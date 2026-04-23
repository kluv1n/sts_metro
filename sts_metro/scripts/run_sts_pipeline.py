#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*pin_memory.*",
    category=UserWarning,
    module=r"torch\.utils\.data\.dataloader",
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_cv.pipeline import PreprocessConfig, load_image, preprocess
from sts_ocr.easyocr_engine import EasyOcrConfig, ocr_easyocr_bgr_with_meta
from sts_ocr.tesseract import TesseractOcrConfig, find_tesseract_executable, ocr_image_bgr_with_meta
from sts_ner.infer import load_ner, predict_entities

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def _list_images(d: Path) -> list[Path]:
    out = []
    for p in sorted(d.iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXT:
            out.append(p)
    return out

def _ocr(
    image_bgr,
    engine: str,
    o_cfg: TesseractOcrConfig,
    easy_cfg: EasyOcrConfig,
) -> dict:
    if engine == "easyocr":
        return ocr_easyocr_bgr_with_meta(image_bgr, easy_cfg)
    return ocr_image_bgr_with_meta(image_bgr, o_cfg)

def main() -> None:
    ap = argparse.ArgumentParser(description="STS: image -> preprocess -> OCR -> NER JSON")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=Path, help="Single image")
    src.add_argument("--input-dir", "-d", type=Path, help="Folder of images")
    ap.add_argument("--ner-model", type=Path, default=ROOT / "source_model_train" / "ner_model")
    ap.add_argument("--out-json", type=Path, default=None, help="Write JSON here (dir: one file per image)")
    ap.add_argument("--profile", choices=("default", "minimal", "aggressive"), default="default")
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--ocr-engine", choices=("easyocr", "tesseract"), default="easyocr")
    ap.add_argument("--easyocr-gpu", action="store_true")
    ap.add_argument("--tesseract", type=Path, default=None, metavar="EXE")
    ap.add_argument("--lang", default="rus+eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--oem", type=int, default=3)
    ap.add_argument("--no-ocr-multipsm", action="store_true")
    args = ap.parse_args()

    if args.tesseract is not None:
        os.environ["TESSERACT_CMD"] = str(Path(args.tesseract).expanduser().resolve())

    if args.ocr_engine == "tesseract" and find_tesseract_executable() is None:
        print("Tesseract not found. Use --ocr-engine easyocr or install tesseract.", file=sys.stderr)
        sys.exit(2)

    p_cfg = PreprocessConfig(max_side=args.max_side)
    o_cfg = TesseractOcrConfig(
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        try_multiple_psm=not args.no_ocr_multipsm,
        scale_min_short_side=1600,
        light_unsharp=True,
    )
    easy_cfg = EasyOcrConfig(gpu=args.easyocr_gpu)

    model, tokenizer, id2label, device = load_ner(args.ner_model.resolve())

    def one_image(path: Path) -> dict:
        img = load_image(path)
        r = preprocess(img, profile=args.profile, config=p_cfg)
        ocr_img = r["image_for_ocr"]
        ocr_out = _ocr(ocr_img, args.ocr_engine, o_cfg, easy_cfg)
        text = ocr_out.get("text") or ""
        entities = predict_entities(text, model, tokenizer, id2label, device)
        return {
            "image": str(path.resolve()),
            "profile": args.profile,
            "ocr_engine": args.ocr_engine,
            "ocr_chars": len(text),
            "ocr_meta": {
                k: ocr_out[k]
                for k in ("mean_conf", "mean_word_conf", "psm", "num_boxes", "langs")
                if k in ocr_out
            },
            "entities": entities,
        }

    if args.input is not None:
        path = args.input.resolve()
        payload = one_image(path)
        s = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(s, encoding="utf-8")
            print(args.out_json)
        else:
            print(s, end="")
        return

    ind = args.input_dir.resolve()
    paths = _list_images(ind)
    if not paths:
        print(f"No images in {ind}", file=sys.stderr)
        sys.exit(1)
    results = [one_image(p) for p in paths]
    if args.out_json:
        out = args.out_json
        if out.suffix.lower() == ".json" and len(paths) > 1:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(out)
        else:
            out.mkdir(parents=True, exist_ok=True)
            for item in results:
                stem = Path(item["image"]).stem
                (out / f"{stem}.pipeline.json").write_text(
                    json.dumps(item, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(out)
    else:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
