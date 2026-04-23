#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_fio.extract import extract_fio_from_ocr_text

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract FIO from 09_ocr.txt under a debug tree")
    ap.add_argument(
        "--ocr-root",
        type=Path,
        required=True,
        help="e.g. debug/ocr_easy (each subfolder should contain 09_ocr.txt)",
    )
    args = ap.parse_args()
    root = args.ocr_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(
        [p.parent for p in root.rglob("09_ocr.txt") if p.parent.is_dir() and not p.parent.name.startswith(".")]
    )
    
    seen: set[Path] = set()
    subs: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        subs.append(p)

    if not subs:
        print(f"No 09_ocr.txt found under {root}", file=sys.stderr)
        sys.exit(1)

    for sub in subs:
        ocr_path = sub / "09_ocr.txt"
        if not ocr_path.is_file():
            print(f"skip {sub.name}: no 09_ocr.txt", file=sys.stderr)
            continue
        text = ocr_path.read_text(encoding="utf-8", errors="replace")
        out = extract_fio_from_ocr_text(text)
        out_path = sub / "10_fio.json"
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{sub.relative_to(root)}: "
            f"ru=({out.get('surname_ru','')} {out.get('name_ru','')} {out.get('patronymic_ru','')}) "
            f"en=({out.get('surname_en','')} {out.get('name_en','')}) "
            f"conf={out.get('confidence')} note={out.get('note')}"
        )
        print(f"  {out_path}")

if __name__ == "__main__":
    main()
