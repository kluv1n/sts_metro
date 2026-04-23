#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_ner.infer import load_ner, predict_entities

def main() -> None:
    ap = argparse.ArgumentParser(description="NER inference: text -> entities JSON")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "source_model_train" / "ner_model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", "-t", type=str, help="Raw OCR text")
    src.add_argument("--text-file", "-f", type=Path, help="UTF-8 file (e.g. 09_ocr.txt)")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8", errors="replace")
    else:
        text = args.text or ""

    model, tokenizer, id2label, device = load_ner(args.model_dir.resolve())
    entities = predict_entities(
        text, model, tokenizer, id2label, device, max_length=args.max_length
    )
    out = {"entities": entities, "chars": len(text)}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
