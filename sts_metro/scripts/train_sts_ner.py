#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_ner.bio import build_label_list
from sts_ner.dataset import StsNerDataset, collate_fn

try:
    import numpy as np
    import torch
    from seqeval.metrics import classification_report, f1_score
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torch.optim import AdamW
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )
except ImportError as e:
    print("Install NER deps: pip install -r requirements-ner.txt", file=sys.stderr)
    raise e

def _load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("samples", []))

def _seqeval_tags_rows(
    preds_rows: list[np.ndarray],
    labels_rows: list[np.ndarray],
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    true_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []
    for pr, lb in zip(preds_rows, labels_rows):
        tt, pp = [], []
        for p, l in zip(pr.tolist(), lb.tolist()):
            if l == -100:
                continue
            tt.append(id2label[int(l)])
            pp.append(id2label[int(p)])
        true_tags.append(tt)
        pred_tags.append(pp)
    return true_tags, pred_tags

def evaluate(
    model,
    loader,
    device,
    id2label: dict[int, str],
) -> dict[str, float]:
    model.eval()
    preds_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits
            preds = logits.argmax(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            attn = batch["attention_mask"].cpu().numpy()
            for bi in range(preds.shape[0]):
                sl = int(attn[bi].sum())
                preds_rows.append(preds[bi, :sl])
                labels_rows.append(labels[bi, :sl])
    true_tags, pred_tags = _seqeval_tags_rows(preds_rows, labels_rows, id2label)
    f1 = f1_score(true_tags, pred_tags)
    return {"token_f1": float(f1)}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", type=Path, default=ROOT / "source_model_train" / "splits" / "train.json")
    ap.add_argument("--val-json", type=Path, default=ROOT / "source_model_train" / "splits" / "val.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "source_model_train" / "ner_model")
    ap.add_argument(
        "--model-name",
        type=str,
        default="DeepPavlov/rubert-base-cased",
        help="HF model id (RuBERT good for Cyrillic; mBERT if Latin-heavy)",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--scheduler",
        choices=("cosine", "linear"),
        default="cosine",
    )
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = build_label_list()
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    train_samples = _load_samples(args.train_json.resolve())
    val_samples = _load_samples(args.val_json.resolve())
    if not train_samples:
        print("No training samples. Run: python scripts/prepare_ner_splits.py", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, fix_mistral_regex=False)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_ds = StsNerDataset(train_samples, tokenizer, label2id, args.max_length)
    val_ds = StsNerDataset(val_samples, tokenizer, label2id, args.max_length)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    collate = partial(collate_fn, pad_token_id=int(pad_id))

    accum = max(1, int(args.grad_accum_steps))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    steps_per_epoch = (len(train_loader) + accum - 1) // accum
    num_steps = steps_per_epoch * args.epochs
    warmup = int(num_steps * args.warmup_ratio)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        sched = get_cosine_schedule_with_warmup(opt, warmup, num_steps)
    else:
        sched = get_linear_schedule_with_warmup(opt, warmup, num_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        opt.zero_grad()
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            total_loss += float(loss.item()) * accum
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            pbar.set_postfix(loss=float(loss.item() * accum))

        metrics = evaluate(model, val_loader, device, id2label)
        f1 = metrics["token_f1"]
        print(f"epoch {epoch+1} train_loss={total_loss/len(train_loader):.4f} val_token_f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1 = f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / "label2id.json").write_text(
                json.dumps(label2id, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  saved best to {args.output_dir}")

    model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
    model.to(device)
    metrics = evaluate(model, val_loader, device, id2label)
    preds_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            pr = out.logits.argmax(-1).cpu().numpy()
            lb = batch["labels"].cpu().numpy()
            attn = batch["attention_mask"].cpu().numpy()
            for bi in range(pr.shape[0]):
                sl = int(attn[bi].sum())
                preds_rows.append(pr[bi, :sl])
                labels_rows.append(lb[bi, :sl])
    true_tags, pred_tags = _seqeval_tags_rows(preds_rows, labels_rows, id2label)
    report = classification_report(true_tags, pred_tags, digits=4)
    (args.output_dir / "val_classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
