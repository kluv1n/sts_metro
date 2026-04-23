
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pytesseract
except ImportError as e:  
    pytesseract = None  
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

@dataclass
class TesseractOcrConfig:
    lang: str = "rus+eng"
    psm: int = 6
    oem: int = 3
    scale_min_short_side: int = 1600
    scale_max_long_side: int = 4200
    light_unsharp: bool = True
    preserve_interword_spaces: bool = True
    try_multiple_psm: bool = True
    candidate_psm: tuple[int, ...] = (3, 4, 6)

_TESSERACT_COMMON_PATHS = (
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/local/bin/tesseract",
)

def _env_tesseract() -> str | None:
    for key in ("TESSERACT_CMD", "TESSERACT_PATH"):
        raw = os.environ.get(key)
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.is_file():
            return str(p)
        if p.is_dir():
            exe = p / "tesseract"
            if exe.is_file():
                return str(exe)
    return None

def _brew_prefix_tesseract() -> str | None:
    brew = shutil.which("brew")
    if not brew:
        return None
    try:
        r = subprocess.run(
            [brew, "--prefix", "tesseract"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    prefix = (r.stdout or "").strip()
    if not prefix:
        return None
    exe = Path(prefix) / "bin" / "tesseract"
    if exe.is_file():
        return str(exe)
    return None

def find_tesseract_executable() -> str | None:
    hit = _env_tesseract()
    if hit:
        return hit
    w = shutil.which("tesseract")
    if w:
        return w
    for cand in _TESSERACT_COMMON_PATHS:
        p = Path(cand)
        if p.is_file():
            return str(p)
    hit = _brew_prefix_tesseract()
    if hit:
        return hit
    return None

def apply_tesseract_cmd() -> str | None:
    if pytesseract is None:
        return None
    exe = find_tesseract_executable()
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    return exe

def tesseract_binary_available() -> bool:
    return find_tesseract_executable() is not None

def _ensure_deps() -> None:
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed. Install with: pip install pytesseract Pillow"
        ) from _IMPORT_ERROR
    if apply_tesseract_cmd() is None:
        raise RuntimeError(
            "Tesseract executable not found (PATH and /opt/homebrew/bin, /usr/local/bin). "
            "Install: macOS `brew install tesseract tesseract-lang`, "
            "Ubuntu `apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng`."
        )

def _prepare_bgr_for_ocr(image_bgr: np.ndarray, cfg: TesseractOcrConfig) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    short, long = min(h, w), max(h, w)
    out = image_bgr
    if cfg.scale_min_short_side > 0 and short < cfg.scale_min_short_side and short > 0:
        scale = cfg.scale_min_short_side / float(short)
        if long * scale > cfg.scale_max_long_side:
            scale = cfg.scale_max_long_side / float(long)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_CUBIC)
    if cfg.light_unsharp:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(g, (0, 0), 1.0)
        sharp = cv2.addWeighted(g, 1.4, blur, -0.4, 0)
        out = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return out

def _tesseract_config_str(cfg: TesseractOcrConfig, psm: int) -> str:
    parts = [f"--oem {cfg.oem}", f"--psm {psm}"]
    if cfg.preserve_interword_spaces:
        parts.append("-c preserve_interword_spaces=1")
    return " ".join(parts)

def _mean_word_confidence_pil(pil, cfg: TesseractOcrConfig, psm: int) -> float:
    custom = _tesseract_config_str(cfg, psm)
    d = pytesseract.image_to_data(
        pil, lang=cfg.lang, config=custom, output_type=pytesseract.Output.DICT
    )
    confs: list[int] = []
    for c in d.get("conf", []) or []:
        try:
            v = int(float(c))
        except (TypeError, ValueError):
            continue
        if v > 0:
            confs.append(v)
    return float(np.mean(confs)) if confs else 0.0

def _ocr_string_pil(pil, cfg: TesseractOcrConfig, psm: int) -> str:
    custom = _tesseract_config_str(cfg, psm)
    text = pytesseract.image_to_string(pil, lang=cfg.lang, config=custom)
    return text if isinstance(text, str) else str(text)

def ocr_image_bgr_with_meta(
    image_bgr: np.ndarray,
    cfg: TesseractOcrConfig | None = None,
) -> dict[str, Any]:
    _ensure_deps()
    cfg = cfg or TesseractOcrConfig()
    work = _prepare_bgr_for_ocr(image_bgr, cfg)
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    from PIL import Image

    pil = Image.fromarray(rgb)

    chosen_psm = cfg.psm
    mean_conf = 0.0
    if cfg.try_multiple_psm and len(cfg.candidate_psm) > 0:
        best_s = -1.0
        for psm in cfg.candidate_psm:
            s = _mean_word_confidence_pil(pil, cfg, psm)
            if s > best_s:
                best_s = s
                chosen_psm = psm
        mean_conf = max(0.0, float(best_s))
    else:
        mean_conf = _mean_word_confidence_pil(pil, cfg, chosen_psm)

    text = _ocr_string_pil(pil, cfg, chosen_psm)
    version = ""
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception:
        pass
    return {
        "text": text,
        "tesseract_version": version,
        "lang": cfg.lang,
        "psm": chosen_psm,
        "mean_word_conf": mean_conf,
        "try_multiple_psm": cfg.try_multiple_psm,
    }

def ocr_image_bgr(
    image_bgr: np.ndarray,
    cfg: TesseractOcrConfig | None = None,
) -> str:
    return ocr_image_bgr_with_meta(image_bgr, cfg)["text"]
