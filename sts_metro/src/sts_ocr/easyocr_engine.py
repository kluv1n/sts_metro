
from __future__ import annotations

import ssl
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

_reader_cache: dict[tuple[str, ...], Any] = {}
_ssl_patched = False

def _patch_default_https_context_for_downloads() -> None:
    global _ssl_patched
    if _ssl_patched:
        return
    try:
        import certifi

        def _ctx() -> ssl.SSLContext:
            return ssl.create_default_context(cafile=certifi.where())

        ssl._create_default_https_context = _ctx  
    except Exception:
        return
    _ssl_patched = True

@dataclass
class EasyOcrConfig:
    langs: tuple[str, ...] = ("ru", "en")
    gpu: bool = False
    min_confidence: float = 0.2

def _get_reader(langs: tuple[str, ...], gpu: bool) -> Any:
    key = langs + (str(gpu),)
    if key not in _reader_cache:
        try:
            import easyocr
        except ImportError as e:  
            raise RuntimeError(
                "easyocr is not installed. Run: pip install -r requirements-ocr-easyocr.txt "
                "(first run downloads model weights, ~100MB+)."
            ) from e
        _patch_default_https_context_for_downloads()
        try:
            _reader_cache[key] = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
        except Exception as e:
            err = str(e).lower()
            if "certificate" in err or "ssl" in err:
                raise RuntimeError(
                    "SSL error while downloading EasyOCR models. Try: "
                    "pip install certifi && re-run; or on macOS open "
                    "`/Applications/Python 3.x/Install Certificates.command`; "
                    "or set SSL_CERT_FILE to your corporate CA bundle."
                ) from e
            raise
    return _reader_cache[key]

def ocr_easyocr_bgr_with_meta(
    image_bgr: np.ndarray,
    cfg: EasyOcrConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or EasyOcrConfig()
    reader = _get_reader(cfg.langs, cfg.gpu)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    parts = reader.readtext(rgb, detail=1, paragraph=False)
    h = float(image_bgr.shape[0])
    line_tol = max(10.0, h * 0.014)

    items: list[tuple[float, float, str]] = []
    for bbox, text, conf in parts:
        t = (text or "").strip()
        if not t or float(conf) < cfg.min_confidence:
            continue
        cy = float(np.mean([float(p[1]) for p in bbox]))
        items.append((cy, float(conf), t))

    items.sort(key=lambda x: x[0])
    if not items:
        return {
            "text": "",
            "engine": "easyocr",
            "mean_conf": 0.0,
            "num_boxes": 0,
            "langs": ",".join(cfg.langs),
        }

    lines_words: list[list[str]] = []
    last_cy = items[0][0]
    cur_words: list[str] = [items[0][2]]
    for cy, _c, t in items[1:]:
        if cy - last_cy <= line_tol:
            cur_words.append(t)
            last_cy = cy
        else:
            lines_words.append(cur_words)
            cur_words = [t]
            last_cy = cy
    lines_words.append(cur_words)

    text = "\n".join(" ".join(w) for w in lines_words)
    mean_conf = float(np.mean([x[1] for x in items]))
    return {
        "text": text,
        "engine": "easyocr",
        "mean_conf": mean_conf,
        "num_boxes": len(items),
        "langs": ",".join(cfg.langs),
    }

__all__ = ["EasyOcrConfig", "ocr_easyocr_bgr_with_meta"]
