#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def _load_all_samples(data_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name == "sts_ocr_annotated.json":
            continue
        if not re.match(r"^file\d+\.json$", p.name, re.I):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for s in data.get("samples", []):
            samples.append(s)
    return samples

def _doc_ids(samples: list[dict]) -> list[str]:
    ids = sorted({s.get("doc_id", "") for s in samples if s.get("doc_id")})
    return [x for x in ids if x]

def _load_exclude_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out

def _resolve_exclude_file(arg_path: Path | None, data_dir: Path) -> tuple[Path | None, bool]:
    if arg_path is not None:
        p = arg_path.expanduser()
        if not p.is_absolute():
            cand = (Path.cwd() / p).resolve()
            if not cand.is_file():
                cand = (ROOT / p).resolve()
            p = cand
        else:
            p = p.resolve()
        return p, True
    default = (data_dir / "holdout_doc_ids.txt").resolve()
    if default.is_file():
        return default, False
    return None, False

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "source_model_train",
        help="Directory with file1.json ... fileN.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <data-dir>/splits",
    )
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude-doc-ids-file",
        type=Path,
        default=None,
        help="One doc_id per line; excluded from train/val/test, written to holdout.json",
    )
    ap.add_argument(
        "--exclude-doc-id-regex",
        type=str,
        default=None,
        help="Optional: drop any doc_id matching this regex (e.g. ^sts_0(0[1-9]|1[0-1])$)",
    )
    ap.add_argument(
        "--list-doc-ids",
        action="store_true",
        help="Print all doc_id from file*.json and exit (for filling holdout_doc_ids.txt)",
    )
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    if args.list_doc_ids:
        raw = _load_all_samples(data_dir)
        ids = sorted({s.get("doc_id", "") for s in raw if s.get("doc_id")})
        for did in ids:
            print(did)
        print(f"# total doc_id: {len(ids)}, total samples: {len(raw)}", file=sys.stderr)
        return

    t, v, te = args.train, args.val, args.test
    if abs(t + v + te - 1.0) > 1e-6:
        print("train+val+test must sum to 1", file=sys.stderr)
        sys.exit(1)

    out_dir = (args.out_dir or (data_dir / "splits")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_samples = _load_all_samples(data_dir)
    if not raw_samples:
        print(f"No samples found in {data_dir} (expected file*.json)", file=sys.stderr)
        sys.exit(1)
    total_before_exclusion = len(raw_samples)

    exclude_file, exclude_explicit = _resolve_exclude_file(args.exclude_doc_ids_file, data_dir)
    if exclude_explicit and not exclude_file.is_file():
        print(
            f"ERROR: --exclude-doc-ids-file not found:\n  {args.exclude_doc_ids_file}\n"
            f"Tried (cwd): {(Path.cwd() / args.exclude_doc_ids_file).resolve()}\n"
            "Create source_model_train/holdout_doc_ids.txt with one doc_id per line "
            "(see: PYTHONPATH=src python scripts/prepare_ner_splits.py --list-doc-ids).",
            file=sys.stderr,
        )
        sys.exit(2)

    exclude_ids = _load_exclude_ids(exclude_file)
    if exclude_file and exclude_ids == set() and exclude_file.is_file():
        msg = f"{exclude_file} has no doc_id lines (only comments). Nothing excluded."
        if exclude_explicit:
            print(f"WARNING: {msg}", file=sys.stderr)
        else:
            print(f"Note: {msg}", file=sys.stderr)
    rx = re.compile(args.exclude_doc_id_regex) if args.exclude_doc_id_regex else None

    holdout: list[dict] = []
    kept: list[dict] = []
    for s in raw_samples:
        did = s.get("doc_id") or ""
        if did in exclude_ids or (rx and rx.match(did)):
            holdout.append(s)
        else:
            kept.append(s)

    samples = kept
    if exclude_ids and not holdout:
        sample_ids = sorted({s.get("doc_id", "") for s in raw_samples if s.get("doc_id")})[:15]
        print(
            "WARNING: exclude list is non-empty but no sample matched.\n"
            f"  Check that doc_id in the file match JSON exactly (e.g. sts_001 vs sts_01).\n"
            f"  First doc_ids in data: {sample_ids}",
            file=sys.stderr,
        )
    if not samples:
        print("After exclusion: no samples left. Check --exclude-doc-ids-file / regex.", file=sys.stderr)
        sys.exit(1)

    doc_ids = _doc_ids(samples)
    rng = random.Random(args.seed)
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(n * t)
    n_val = int(n * v)
    n_test = n - n_train - n_val
    train_ids = set(doc_ids[:n_train])
    val_ids = set(doc_ids[n_train : n_train + n_val])
    test_ids = set(doc_ids[n_train + n_val :])

    def pick(id_set: set[str]) -> list[dict]:
        return [s for s in samples if s.get("doc_id") in id_set]

    train_s = pick(train_ids)
    val_s = pick(val_ids)
    test_s = pick(test_ids)

    meta = {
        "seed": args.seed,
        "total_samples_before_exclusion": total_before_exclusion,
        "holdout_samples": len(holdout),
        "exclude_doc_ids_file": str(exclude_file) if exclude_file else None,
        "exclude_doc_ids_explicit_arg": exclude_explicit,
        "exclude_doc_id_regex": args.exclude_doc_id_regex,
        "total_samples": len(samples),
        "total_doc_ids": n,
        "train_docs": len(train_ids),
        "val_docs": len(val_ids),
        "test_docs": len(test_ids),
        "train_samples": len(train_s),
        "val_samples": len(val_s),
        "test_samples": len(test_s),
    }
    (out_dir / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, arr in ("train", train_s), ("val", val_s), ("test", test_s):
        (out_dir / f"{name}.json").write_text(
            json.dumps({"samples": arr}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (out_dir / "holdout.json").write_text(
        json.dumps({"samples": holdout}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_dir / 'train.json'}")
    print(f"Wrote: {out_dir / 'val.json'}")
    print(f"Wrote: {out_dir / 'test.json'}")
    print(f"Wrote: {out_dir / 'holdout.json'} ({len(holdout)} samples)")

if __name__ == "__main__":
    main()
