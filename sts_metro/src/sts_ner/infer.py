
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sts_ner.bio import labels_to_entity_strings

def load_ner(model_dir: Path, device: torch.device | None = None):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_dir = Path(model_dir).resolve()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, fix_mistral_regex=False)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.to(dev)
    model.eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    return model, tok, id2label, dev

def predict_entities(
    text: str,
    model: Any,
    tokenizer: Any,
    id2label: dict[int, str],
    device: torch.device,
    *,
    max_length: int = 512,
) -> dict[str, str]:
    raw = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        padding=False,
    )
    offsets = raw["offset_mapping"]
    batch = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(**batch).logits
    pred = logits.argmax(-1)[0].cpu().numpy().tolist()

    m = min(len(pred), len(offsets))
    return labels_to_entity_strings(text, offsets[:m], pred[:m], id2label)
