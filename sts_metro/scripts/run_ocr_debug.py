#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import cv2

warnings.filterwarnings(
    "ignore",
    message=r".*pin_memory.*",
    category=UserWarning,
    module=r"torch\.utils\.data\.dataloader",
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_cv.pipeline import PreprocessConfig, load_image, preprocess, save_debug_steps
from sts_ocr.easyocr_engine import EasyOcrConfig, ocr_easyocr_bgr_with_meta
from sts_ocr.tesseract import (
    TesseractOcrConfig,
    find_tesseract_executable,
    ocr_image_bgr_with_meta,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files: list[Path] = []
    for p in directory.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())

def _binary_ocr_variant(image_bgr, invert: bool = False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    if invert:
        bw = cv2.bitwise_not(bw)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def _detect_text_line_bands(binary_bgr):
    gray = cv2.cvtColor(binary_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 128).astype("uint8")
    light_mask = (gray > 127).astype("uint8")
    
    text_mask = dark_mask if dark_mask.sum() <= light_mask.sum() else light_mask

    h, w = text_mask.shape
    proj = text_mask.sum(axis=1)
    row_thr = max(4, int(w * 0.006))

    runs = []
    y = 0
    while y < h:
        if proj[y] >= row_thr:
            y0 = y
            while y < h and proj[y] >= row_thr:
                y += 1
            y1 = y
            if (y1 - y0) >= 4:
                runs.append((y0, y1))
        else:
            y += 1

    if runs:
        heights = sorted(y1 - y0 for y0, y1 in runs)
        median_h = heights[len(heights) // 2]
    else:
        median_h = max(10, h // 40)

    band_h = max(26, min(72, int(median_h * 3.0)))
    
    stride = max(14, int(band_h * 0.78))

    out = []
    y0 = 0
    while y0 < h:
        y1 = min(h, y0 + band_h)
        out.append((y0, y1))
        if y1 >= h:
            break
        y0 += stride
    if out and out[-1][1] < h:
        out.append((max(0, h - band_h), h))

    out = out[:64]
    return out

def _ocr_by_line_bands(ocr_image, bands, ocr_engine, o_cfg, easy_cfg):
    lines = []
    confs = []
    if ocr_engine == "tesseract":
        line_cfg = replace(o_cfg, psm=7, try_multiple_psm=False)
    else:
        line_cfg = None

    for y0, y1 in bands:
        crop = ocr_image[y0:y1, :]
        if crop.size == 0:
            continue
        if ocr_engine == "easyocr":
            out = ocr_easyocr_bgr_with_meta(crop, easy_cfg)
            txt = (out.get("text") or "").strip()
            confs.append(float(out.get("mean_conf", 0.0)))
        else:
            out = ocr_image_bgr_with_meta(crop, line_cfg)
            txt = (out.get("text") or "").strip()
            confs.append(float(out.get("mean_word_conf", 0.0)))
        if txt:
            lines.append(txt)

    text = "\n".join(lines).strip()
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    return {"text": text, "num_lines": len(lines), "mean_line_conf": mean_conf}

def _run_one_pass(
    input_path: Path,
    out_dir: Path,
    p_cfg: PreprocessConfig,
    profile: str,
    ocr_engine: str,
    o_cfg: TesseractOcrConfig,
    easy_cfg: EasyOcrConfig,
    skip_ocr: bool,
    variant: str,
    bw_line_ocr: bool,
) -> None:
    img = load_image(input_path)
    result = preprocess(img, profile=profile, config=p_cfg)
    paths = save_debug_steps(result["steps"], out_dir)

    ocr_image = result["image_for_ocr"]
    if variant == "bw":
        bw = _binary_ocr_variant(ocr_image, invert=False)
        p_bw = out_dir / "06b_for_ocr_binary_bw.png"
        cv2.imwrite(str(p_bw), bw)
        paths.append(p_bw)
        ocr_image = bw
    elif variant == "bw_inv":
        bw_inv = _binary_ocr_variant(ocr_image, invert=True)
        p_bw_inv = out_dir / "06c_for_ocr_binary_bw_inverted.png"
        cv2.imwrite(str(p_bw_inv), bw_inv)
        paths.append(p_bw_inv)
        ocr_image = bw_inv

    alt = result.get("image_for_ocr_alt")
    if alt is not None and variant == "default":
        p = out_dir / "08_for_ocr_aggressive_alt.png"
        cv2.imwrite(str(p), alt)
        paths.append(p)

    line_bands = []
    if bw_line_ocr and variant in {"bw", "bw_inv"}:
        line_bands = _detect_text_line_bands(ocr_image)
        if line_bands:
            overlay = ocr_image.copy()
            h, w = overlay.shape[:2]
            for i, (y0, y1) in enumerate(line_bands, start=1):
                cv2.rectangle(overlay, (0, y0), (w - 1, y1 - 1), (0, 255, 0), 1)
                cv2.putText(
                    overlay,
                    str(i),
                    (4, max(12, y0 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            p_bands = out_dir / "09_line_bands.png"
            cv2.imwrite(str(p_bands), overlay)
            paths.append(p_bands)

    ocr_path = out_dir / "09_ocr.txt"
    if skip_ocr:
        ocr_path.write_text("", encoding="utf-8")
        meta_line = "ocr=skipped"
    else:
        if line_bands:
            ocr_out = _ocr_by_line_bands(ocr_image, line_bands, ocr_engine, o_cfg, easy_cfg)
            (out_dir / "09_ocr_meta.txt").write_text(
                f"engine={ocr_engine}\n"
                f"line_ocr=true\n"
                f"line_bands={len(line_bands)}\n"
                f"line_text_rows={ocr_out.get('num_lines', 0)}\n"
                f"mean_line_conf={ocr_out.get('mean_line_conf', 0):.2f}\n",
                encoding="utf-8",
            )
            meta_line = (
                f"ocr={ocr_engine} line_ocr=1 chars={len(ocr_out['text'])} "
                f"bands={len(line_bands)} mean_line_conf={ocr_out.get('mean_line_conf', 0):.1f}"
            )
        else:
            if ocr_engine == "easyocr":
                ocr_out = ocr_easyocr_bgr_with_meta(ocr_image, easy_cfg)
                (out_dir / "09_ocr_meta.txt").write_text(
                    f"engine=easyocr\n"
                    f"mean_conf={ocr_out.get('mean_conf', 0):.2f}\n"
                    f"boxes={ocr_out.get('num_boxes', 0)}\n"
                    f"langs={ocr_out.get('langs', '')}\n",
                    encoding="utf-8",
                )
                meta_line = (
                    f"ocr=easyocr chars={len(ocr_out['text'])} "
                    f"mean_conf={ocr_out.get('mean_conf', 0):.1f} boxes={ocr_out.get('num_boxes', 0)}"
                )
            else:
                ocr_out = ocr_image_bgr_with_meta(ocr_image, o_cfg)
                (out_dir / "09_ocr_meta.txt").write_text(
                    f"engine=tesseract\n"
                    f"psm={ocr_out['psm']}\n"
                    f"mean_word_conf={ocr_out.get('mean_word_conf', 0):.2f}\n"
                    f"multipsm={ocr_out.get('try_multiple_psm', False)}\n",
                    encoding="utf-8",
                )
                meta_line = (
                    f"ocr=tesseract chars={len(ocr_out['text'])} "
                    f"psm={ocr_out['psm']} mean_conf={ocr_out.get('mean_word_conf', 0):.1f}"
                )
        ocr_path.write_text(ocr_out["text"], encoding="utf-8")

    print(
        f"{input_path.name} [{variant}]: postQ={result.get('post_deskew_quarter_ccw', 0)} "
        f"deskew={result.get('deskew_deg', 0):.2f} {meta_line}"
    )
    print(f"  {ocr_path}")
    if not skip_ocr:
        print(f"  {out_dir / '09_ocr_meta.txt'}")
    for pth in paths:
        print(f"  {pth}")

def main() -> None:
    ap = argparse.ArgumentParser(description="STS CV + OCR (Tesseract or EasyOCR)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", type=Path, help="One input image")
    src.add_argument("--input-dir", "-d", type=Path, help="Folder of images → subfolders under --out")
    ap.add_argument("--out", "-o", type=Path, default=Path("debug/ocr"))
    ap.add_argument("--profile", choices=("default", "minimal", "aggressive"), default="default")
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument("--rotate", type=float, default=0.0)
    ap.add_argument("--no-warp", action="store_true")
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--no-orient", action="store_true")
    ap.add_argument("--bbox-crop", action="store_true")
    ap.add_argument("--force-180", action="store_true")
    ap.add_argument("--no-upright-180", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true", help="Only CV debug steps, write empty 09_ocr.txt")
    ap.add_argument(
        "--tesseract",
        type=Path,
        metavar="EXE",
        help="Path to tesseract binary (or set env TESSERACT_CMD)",
    )
    ap.add_argument("--lang", default="rus+eng", help="Tesseract -l (e.g. rus+eng)")
    ap.add_argument(
        "--psm",
        type=int,
        default=6,
        help="PSM when --no-ocr-multipsm; else fallback if multipsm disabled",
    )
    ap.add_argument("--oem", type=int, default=3, help="Tesseract OCR engine mode")
    ap.add_argument(
        "--ocr-engine",
        choices=("tesseract", "easyocr"),
        default="tesseract",
        help="OCR backend (easyocr: pip install -r requirements-ocr-easyocr.txt)",
    )
    ap.add_argument(
        "--no-ocr-multipsm",
        action="store_true",
        help="Tesseract only: do not try PSM 3,4,6; use single --psm",
    )
    ap.add_argument("--easyocr-gpu", action="store_true", help="EasyOCR only: use GPU if available")
    ap.add_argument("--no-ocr-upscale", action="store_true", help="Disable OCR-only upscale for small text (Tesseract)")
    ap.add_argument("--no-ocr-unsharp", action="store_true", help="Disable mild unsharp before OCR (Tesseract)")
    ap.add_argument(
        "--no-bw-pass",
        action="store_true",
        help="Disable additional OCR pass on pure black/white image",
    )
    ap.add_argument(
        "--no-bw-inv-pass",
        action="store_true",
        help="Disable additional OCR pass on inverted pure black/white image",
    )
    ap.add_argument(
        "--bw-line-ocr",
        action="store_true",
        help="Run OCR by simple line bands for bw and bw_inv variants",
    )
    args = ap.parse_args()

    if args.tesseract is not None:
        exe = Path(args.tesseract).expanduser()
        if not exe.is_file():
            print(f"ERROR: --tesseract is not a file: {exe}", file=sys.stderr)
            sys.exit(2)
        os.environ["TESSERACT_CMD"] = str(exe)

    if not args.skip_ocr and args.ocr_engine == "tesseract" and find_tesseract_executable() is None:
        print(
            "ERROR: Tesseract not found. Try:\n"
            "  brew install tesseract tesseract-lang\n"
            "  export TESSERACT_CMD=\"$(brew --prefix tesseract)/bin/tesseract\"\n"
            "Or use: --ocr-engine easyocr   Or: --skip-ocr",
            file=sys.stderr,
        )
        sys.exit(2)

    p_cfg = PreprocessConfig(
        max_side=args.max_side,
        rotate_deg=args.rotate,
        try_perspective_warp=not args.no_warp,
        auto_deskew=not args.no_deskew,
        auto_page_orientation=not args.no_orient,
        fallback_bbox_crop=args.bbox_crop,
        force_rotate_180=args.force_180,
        auto_upright_180=not args.no_upright_180,
    )
    o_cfg = TesseractOcrConfig(
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        try_multiple_psm=not args.no_ocr_multipsm,
        scale_min_short_side=0 if args.no_ocr_upscale else 1600,
        light_unsharp=not args.no_ocr_unsharp,
    )
    easy_cfg = EasyOcrConfig(gpu=args.easyocr_gpu)
    out_root = Path(args.out)

    do_bw = not args.no_bw_pass
    do_bw_inv = not args.no_bw_inv_pass
    out_default = out_root / "default"
    out_bw = out_root / "bw"
    out_bw_inv = out_root / "bw_inv"

    if args.input is not None:
        out_default.mkdir(parents=True, exist_ok=True)
        _run_one_pass(
            args.input.resolve(),
            out_default,
            p_cfg,
            args.profile,
            args.ocr_engine,
            o_cfg,
            easy_cfg,
            args.skip_ocr,
            variant="default",
            bw_line_ocr=args.bw_line_ocr,
        )
        if do_bw:
            out_bw.mkdir(parents=True, exist_ok=True)
            _run_one_pass(
                args.input.resolve(),
                out_bw,
                p_cfg,
                args.profile,
                args.ocr_engine,
                o_cfg,
                easy_cfg,
                args.skip_ocr,
                variant="bw",
                bw_line_ocr=args.bw_line_ocr,
            )
        if do_bw_inv:
            out_bw_inv.mkdir(parents=True, exist_ok=True)
            _run_one_pass(
                args.input.resolve(),
                out_bw_inv,
                p_cfg,
                args.profile,
                args.ocr_engine,
                o_cfg,
                easy_cfg,
                args.skip_ocr,
                variant="bw_inv",
                bw_line_ocr=args.bw_line_ocr,
            )
        return

    paths = _list_images(args.input_dir.resolve())
    if not paths:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    out_default.mkdir(parents=True, exist_ok=True)
    if do_bw:
        out_bw.mkdir(parents=True, exist_ok=True)
    if do_bw_inv:
        out_bw_inv.mkdir(parents=True, exist_ok=True)
    for inp in paths:
        sub_default = out_default / inp.stem
        _run_one_pass(
            inp,
            sub_default,
            p_cfg,
            args.profile,
            args.ocr_engine,
            o_cfg,
            easy_cfg,
            args.skip_ocr,
            variant="default",
            bw_line_ocr=args.bw_line_ocr,
        )
        if do_bw:
            sub_bw = out_bw / inp.stem
            _run_one_pass(
                inp,
                sub_bw,
                p_cfg,
                args.profile,
                args.ocr_engine,
                o_cfg,
                easy_cfg,
                args.skip_ocr,
                variant="bw",
                bw_line_ocr=args.bw_line_ocr,
            )
        if do_bw_inv:
            sub_bw_inv = out_bw_inv / inp.stem
            _run_one_pass(
                inp,
                sub_bw_inv,
                p_cfg,
                args.profile,
                args.ocr_engine,
                o_cfg,
                easy_cfg,
                args.skip_ocr,
                variant="bw_inv",
                bw_line_ocr=args.bw_line_ocr,
            )

if __name__ == "__main__":
    main()
