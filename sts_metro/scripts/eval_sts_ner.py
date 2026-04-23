#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_ner.bio import ENTITY_KEYS, build_label_list, labels_to_entity_strings
from sts_ner.dataset import StsNerDataset, collate_fn

try:
    import numpy as np
    import torch
    from seqeval.metrics import classification_report, f1_score
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError as e:
    print("Install: pip install -r requirements-ner.txt", file=sys.stderr)
    raise e

def _load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("samples", []))

def _norm_entity(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "")

def _load_tokenizer(model_dir: Path):
    return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, fix_mistral_regex=False)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=ROOT / "source_model_train" / "ner_model")
    ap.add_argument("--test-json", type=Path, default=ROOT / "source_model_train" / "splits" / "test.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    test_path = args.test_json.resolve()
    if not test_path.is_file():
        print(f"ERROR: --test-json not found:\n  {test_path}", file=sys.stderr)
        sys.exit(2)
    samples = _load_samples(test_path)
    if not samples:
        print("No samples in JSON (empty list). Nothing to evaluate.")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    labels = build_label_list()
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    ds = StsNerDataset(samples, tokenizer, label2id, args.max_length)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=int(pad_id)),
    )

    preds_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    field_hits = {k: 0 for k in ENTITY_KEYS}
    field_total = {k: 0 for k in ENTITY_KEYS}
    global_i = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            bsz = batch["input_ids"].shape[0]
            batch_d = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch_d).logits
            preds = logits.argmax(-1).cpu().numpy()
            labels_np = batch["labels"].cpu().numpy()
            attn = batch["attention_mask"].cpu().numpy()

            for bi in range(bsz):
                text = samples[global_i]["text"]
                seq_len = int(attn[bi].sum())
                pr = preds[bi][:seq_len].tolist()
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=args.max_length,
                    return_offsets_mapping=True,
                )
                offsets = enc["offset_mapping"][:seq_len]
                m = min(len(pr), len(offsets))
                pred_ent = labels_to_entity_strings(text, offsets[:m], pr[:m], id2label)
                gold = samples[global_i].get("entities") or {}
                for k in ENTITY_KEYS:
                    g = _norm_entity(str(gold.get(k, "")))
                    if not g:
                        continue
                    field_total[k] += 1
                    p = _norm_entity(str(pred_ent.get(k, "")))
                    if p == g:
                        field_hits[k] += 1
                preds_rows.append(preds[bi, :seq_len])
                labels_rows.append(labels_np[bi, :seq_len])
                global_i += 1

    tt, pp = [], []
    for pr, lb in zip(preds_rows, labels_rows):
        row_t, row_p = [], []
        for p, l in zip(pr.tolist(), lb.tolist()):
            if l == -100:
                continue
            row_t.append(id2label[int(l)])
            row_p.append(id2label[int(p)])
        tt.append(row_t)
        pp.append(row_p)
    f1 = f1_score(tt, pp)
    report = classification_report(tt, pp, digits=4)

    out_dir = args.model_dir
    (out_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "token_f1": f1,
                "field_recall_exact_when_gold_present": {
                    k: (field_hits[k] / field_total[k] if field_total[k] else None)
                    for k in ENTITY_KEYS
                },
                "field_hits": field_hits,
                "field_total_gold_present": field_total,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "test_classification_report.txt").write_text(report, encoding="utf-8")

    print(f"token_f1 (seqeval): {f1:.4f}")
    print("per-field exact when gold non-empty (normalized):")
    for k in ENTITY_KEYS:
        t = field_total[k]
        print(f"  {k}: {field_hits[k]}/{t} = {field_hits[k]/t if t else 0:.4f}")
    print(report)
    print(f"Wrote {out_dir / 'test_metrics.json'}")

if __name__ == "__main__":
    main()
