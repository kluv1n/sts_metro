#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_cv.pipeline import PreprocessConfig, load_image, preprocess, save_debug_steps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    files: list[Path] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())

def _process_one(
    input_path: Path,
    out_dir: Path,
    cfg: PreprocessConfig,
    profile: str,
) -> None:
    img = load_image(input_path)
    result = preprocess(img, profile=profile, config=cfg)
    paths = save_debug_steps(result["steps"], out_dir)
    alt = result.get("image_for_ocr_alt")
    if alt is not None:
        p = out_dir / "08_for_ocr_aggressive_alt.png"
        cv2.imwrite(str(p), alt)
        paths.append(p)

    print(
        f"{input_path.name}: warp={result['warp_applied']} "
        f"bbox={result.get('bbox_crop_applied', False)} trim={result.get('border_trim_applied', False)} "
        f"orient_k={result.get('page_orient_quarter_ccw', 0)} flip180={result.get('page_orient_flip_180', 0)} "
        f"orient_skip={result.get('page_orient_skipped', False)} "
        f"E={result.get('page_orient_energy', 0):.0f} "
        f"deskew={result.get('deskew_deg', 0):.2f} rot={result.get('rotation_applied_deg', 0):.2f} "
        f"postQ={result.get('post_deskew_quarter_ccw', 0)} "
        f"tb_desk={result.get('post_deskew_tb_ratio_deskewed', 0):.2f} "
        f"ud_sig={result.get('post_deskew_upside_down_signal', False)} "
        f"res180={result.get('post_deskew_180_resolution', '')} "
        f"tb_out={result.get('upright_180_row_var_ratio', 0):.2f} "
        f"profile={profile}"
    )
    for pth in paths:
        print(f"  {pth}")

def main() -> None:
    ap = argparse.ArgumentParser(description="STS CV preprocess debug dump")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", help="Path to one input image")
    src.add_argument(
        "--input-dir",
        "-d",
        type=Path,
        help="Process all images in this folder (each → subfolder under --out)",
    )
    ap.add_argument(
        "--out",
        "-o",
        default="debug/cv",
        help="Output directory: one folder for single -i, or parent for -d",
    )
    ap.add_argument(
        "--profile",
        choices=("default", "minimal", "aggressive"),
        default="default",
    )
    ap.add_argument("--max-side", type=int, default=2000)
    ap.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="Extra rotation in degrees after auto-deskew (CCW positive in OpenCV)",
    )
    ap.add_argument("--no-warp", action="store_true", help="Disable perspective warp")
    ap.add_argument("--no-deskew", action="store_true", help="Disable automatic deskew")
    ap.add_argument("--no-orient", action="store_true", help="Disable 0/90/180/270 page orientation search")
    ap.add_argument("--bbox-crop", action="store_true", help="Enable risky bbox crop when warp fails")
    ap.add_argument(
        "--force-180",
        action="store_true",
        help="Rotate 180° after auto-orientation (manual override)",
    )
    ap.add_argument(
        "--no-upright-180",
        action="store_true",
        help="Disable auto 180° after deskew (row-variance top/bottom heuristic)",
    )
    args = ap.parse_args()

    cfg = PreprocessConfig(
        max_side=args.max_side,
        rotate_deg=args.rotate,
        try_perspective_warp=not args.no_warp,
        auto_deskew=not args.no_deskew,
        auto_page_orientation=not args.no_orient,
        fallback_bbox_crop=args.bbox_crop,
        force_rotate_180=args.force_180,
        auto_upright_180=not args.no_upright_180,
    )
    out_root = Path(args.out)

    if args.input is not None:
        out_root.mkdir(parents=True, exist_ok=True)
        _process_one(Path(args.input), out_root, cfg, args.profile)
        return

    paths = _list_images(args.input_dir.resolve())
    if not paths:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    for inp in paths:
        sub = out_root / inp.stem
        _process_one(inp, sub, cfg, args.profile)

if __name__ == "__main__":
    main()
