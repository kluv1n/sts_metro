#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sts_sort_key(name: str) -> tuple[int, str]:
    m = re.match(r"^sts_(\d+)$", name)
    if m:
        return (int(m.group(1)), name)
    return (9999, name)


def _ner_index(batch: dict) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for it in batch.get("items", []):
        rel = it.get("json") or ""
        parts = Path(rel).parts
        if len(parts) < 4:
            continue
        if parts[0] != "debug" or parts[1] != "ocr_compare":
            continue
        variant, sts = parts[2], parts[3]
        if not sts.startswith("sts_"):
            continue
        out[(variant, sts)] = it
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-compare", type=Path, default=ROOT / "debug" / "ocr_compare")
    ap.add_argument("--ner-batch", type=Path, default=ROOT / "debug" / "ner_batch_output.json")
    ap.add_argument("--fio-name", type=str, default="10_fio.json")
    ap.add_argument("--out-json", type=Path, default=ROOT / "debug" / "sts_fio_ner_merged.json")
    ap.add_argument(
        "--per-sts-dir",
        type=Path,
        default=ROOT / "debug" / "sts_merged",
    )
    ap.add_argument("--no-per-sts-files", action="store_true")
    args = ap.parse_args()

    ocr_root = args.ocr_compare.resolve()
    if not ocr_root.is_dir():
        print(f"Not a directory: {ocr_root}", file=sys.stderr)
        sys.exit(2)

    ner_path = args.ner_batch.resolve()
    if not ner_path.is_file():
        print(f"Missing NER batch file: {ner_path}", file=sys.stderr)
        sys.exit(2)

    batch = json.loads(ner_path.read_text(encoding="utf-8"))
    ner_by = _ner_index(batch)

    sts_set: set[str] = set()
    variant_set: set[str] = set()
    for (variant, sts) in ner_by:
        sts_set.add(sts)
        variant_set.add(variant)
    for vd in ocr_root.iterdir():
        if not vd.is_dir():
            continue
        variant_set.add(vd.name)
        for sd in vd.iterdir():
            if sd.is_dir() and sd.name.startswith("sts_"):
                sts_set.add(sd.name)

    sts_ids = sorted(sts_set, key=_sts_sort_key)
    variant_names = sorted(variant_set)

    per_sts: dict[str, dict[str, dict]] = {}
    for sts in sts_ids:
        by_var: dict[str, dict] = {}
        for variant in variant_names:
            fio_path = ocr_root / variant / sts / args.fio_name
            rule_obj: dict | None = None
            if fio_path.is_file():
                rule_obj = json.loads(fio_path.read_text(encoding="utf-8"))
            ner_item = ner_by.get((variant, sts))
            ner_block: dict = {}
            if ner_item:
                ner_block = {
                    "ocr_source": ner_item.get("ocr_source"),
                    "ocr_chars": ner_item.get("ocr_chars"),
                    "entities": ner_item.get("entities") or {},
                }
                if ner_item.get("error"):
                    ner_block["error"] = ner_item["error"]
            by_var[variant] = {"rule_fio": rule_obj, "ner": ner_block}
        per_sts[sts] = by_var

    out_path = args.out_json.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ocr_compare_root": str(ocr_root.relative_to(ROOT)),
        "ner_batch_source": str(ner_path.relative_to(ROOT)),
        "fio_json_name": args.fio_name,
        "variant_names": variant_names,
        "sts_ids": sts_ids,
        "per_sts": per_sts,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_extra = 0
    ps = args.per_sts_dir
    if not args.no_per_sts_files and ps is not None:
        ps = ps.resolve()
        ps.mkdir(parents=True, exist_ok=True)
        for sts in sts_ids:
            one = {
                "sts_id": sts,
                "ocr_compare_root": payload["ocr_compare_root"],
                "ner_batch_source": payload["ner_batch_source"],
                "fio_json_name": args.fio_name,
                "variant_names": variant_names,
                "variants": per_sts[sts],
            }
            (ps / f"{sts}.json").write_text(
                json.dumps(one, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            n_extra += 1
    print(f"Wrote {out_path} ({len(sts_ids)} sts × {len(variant_names)} variants)")
    if n_extra:
        print(f"Wrote {n_extra} files under {ps}")


if __name__ == "__main__":
    main()
