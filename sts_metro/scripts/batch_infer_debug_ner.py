#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_ner.infer import load_ner, predict_entities


def _ocr_text_for_json(json_path: Path) -> tuple[str | None, str | None]:
    parent = json_path.parent
    preferred = parent / "09_ocr.txt"
    if preferred.is_file():
        return preferred.read_text(encoding="utf-8", errors="replace"), str(preferred.relative_to(ROOT))
    cands = sorted(parent.glob("*_ocr.txt"))
    if cands:
        p = cands[0]
        return p.read_text(encoding="utf-8", errors="replace"), str(p.relative_to(ROOT))
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--debug-root",
        type=Path,
        default=ROOT / "debug" / "ocr_compare",
        help="Root for recursive *.json (default: debug/ocr_compare)",
    )
    ap.add_argument("--model-dir", type=Path, default=ROOT / "source_model_train" / "ner_model")
    ap.add_argument("--out-json", type=Path, default=ROOT / "debug" / "ner_batch_output.json")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    root = args.debug_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    model_dir = args.model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        print(f"Missing model config: {model_dir}", file=sys.stderr)
        sys.exit(2)

    json_paths = sorted(root.rglob("*.json"))
    if not json_paths:
        print(f"No JSON under {root}", file=sys.stderr)
        sys.exit(1)

    model, tokenizer, id2label, device = load_ner(model_dir)
    text_cache: dict[Path, tuple[str | None, str | None]] = {}
    rows: list[dict] = []

    for jp in json_paths:
        rel = str(jp.relative_to(ROOT))
        key = jp.parent.resolve()
        if key not in text_cache:
            text_cache[key] = _ocr_text_for_json(jp)
        text, ocr_rel = text_cache[key]
        if not text:
            rows.append(
                {
                    "json": rel,
                    "ocr_source": None,
                    "error": "no_*_ocr.txt_in_parent",
                    "entities": {},
                }
            )
            continue
        entities = predict_entities(
            text,
            model,
            tokenizer,
            id2label,
            device,
            max_length=args.max_length,
        )
        rows.append(
            {
                "json": rel,
                "ocr_source": ocr_rel,
                "ocr_chars": len(text),
                "entities": entities,
            }
        )

    out_path = args.out_json.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"debug_root": str(root), "model_dir": str(model_dir), "count": len(rows), "items": rows}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} JSON files)")


if __name__ == "__main__":
    main()
