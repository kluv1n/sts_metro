from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

Profile = Literal["default", "minimal", "aggressive"]

@dataclass
class PreprocessConfig:
    max_side: int = 2000
    rotate_deg: float = 0.0
    auto_page_orientation: bool = True
    page_orient_portrait_skip_high_ratio: float = 0.93
    page_orient_portrait_skip_low_ratio: float = 0.49
    auto_deskew: bool = True
    deskew_use_center_crop_frac: tuple[float, float, float, float] = (0.06, 0.94, 0.06, 0.94)
    deskew_angle_range_deg: tuple[float, float] = (-7.0, 7.0)
    deskew_step_deg: float = 0.25
    deskew_refine_radius_deg: float = 1.2
    deskew_refine_step_deg: float = 0.05
    deskew_extend_angle_range_deg: tuple[float, float] = (-18.0, 18.0)
    deskew_weak_extend_step_deg: float = 1.0
    deskew_weak_score_ratio_max: float = 1.06
    auto_upright_180: bool = True
    upright_180_ratio_threshold: float = 1.95
    post_deskew_portrait_h_over_w_min: float = 1.01
    post_deskew_quarter_energy_floor_ratio: float = 0.998
    post_deskew_180_energy_prefer_k2_margin: float = 0.01
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    denoise_h: int = 8
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    try_perspective_warp: bool = True
    warp_max_side_for_search: int = 900
    canny_low: int = 40
    canny_high: int = 140
    contour_min_area_ratio: float = 0.12
    warp_max_area_ratio: float = 0.88
    warp_min_output_area_ratio: float = 0.18
    warp_max_aspect_ratio: float = 2.85
    fallback_bbox_crop: bool = False
    bbox_crop_min_area_ratio: float = 0.2
    bbox_crop_pad_ratio: float = 0.02
    trim_border_max_frac: float = 0.0
    trim_border_step_px: int = 2
    trim_border_mean_thresh: float = 18.0
    auto_sts_red_upright: bool = False
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    force_rotate_180: bool = False

def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {p}")
    return img

def _resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def _rotate(bgr: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-3:
        return bgr
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(
        bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

def _center_crop_gray(gray: np.ndarray, frac: tuple[float, float, float, float]) -> np.ndarray:
    y0f, y1f, x0f, x1f = frac
    h, w = gray.shape[:2]
    y0, y1 = int(h * y0f), int(h * y1f)
    x0, x1 = int(w * x0f), int(w * x1f)
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 <= y0 + 8 or x1 <= x0 + 8:
        return gray
    return gray[y0:y1, x0:x1]

def _max_line_energy_all_orientations(bgr: np.ndarray) -> float:
    mx = 0.0
    for k in range(4):
        cur = bgr
        for _ in range(k):
            cur = cv2.rotate(cur, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for f in (0, 1):
            cand = cv2.rotate(cur, cv2.ROTATE_180) if f else cur
            mx = max(mx, _horizontal_line_energy(cand))
    return mx

def _horizontal_line_energy(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = _center_crop_gray(gray, (0.05, 0.95, 0.05, 0.95))
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = np.sum(b.astype(np.float64), axis=1)
    d = np.diff(proj)
    return float(np.sum(d * d))

def _pick_page_orientation(
    bgr: np.ndarray, cfg: PreprocessConfig
) -> tuple[np.ndarray, int, int, float, bool]:
    if not cfg.auto_page_orientation:
        e0 = _horizontal_line_energy(bgr)
        return bgr, 0, 0, e0, False

    h, w = bgr.shape[:2]
    landscape = w > h * 1.02
    portrait = h > w * 1.02
    le0 = _horizontal_line_energy(bgr)
    mx = _max_line_energy_all_orientations(bgr)
    ratio = le0 / max(mx, 1e-6)

    skip_search = False
    if portrait and not landscape:
        if ratio >= cfg.page_orient_portrait_skip_high_ratio:
            skip_search = True
        if ratio <= cfg.page_orient_portrait_skip_low_ratio:
            skip_search = True

    if skip_search:
        return bgr, 0, 0, le0, True

    best = bgr
    best_k = 0
    best_f = 0
    best_e = -1.0
    for k in range(4):
        cur = bgr
        for _ in range(k):
            cur = cv2.rotate(cur, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for f in (0, 1):
            cand = cv2.rotate(cur, cv2.ROTATE_180) if f else cur
            e = _horizontal_line_energy(cand)
            if e > best_e:
                best_e = e
                best = cand
                best_k = k
                best_f = f
    return best, best_k, best_f, best_e, False

def _trim_uniform_edges(bgr: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, bool]:
    if cfg.trim_border_max_frac <= 0:
        return bgr, False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    lim_y = int(h * cfg.trim_border_max_frac)
    lim_x = int(w * cfg.trim_border_max_frac)
    t, b, l, r = 0, h - 1, 0, w - 1
    th = cfg.trim_border_mean_thresh
    step = cfg.trim_border_step_px

    def row_mean(y: int) -> float:
        return float(np.mean(gray[y, :]))

    def col_mean(x: int) -> float:
        return float(np.mean(gray[:, x]))

    while t + step < h - 16 and t < lim_y and row_mean(t) < th:
        t += step
    while b - step > t + 16 and h - 1 - b < lim_y and row_mean(b) < th:
        b -= step
    while l + step < w - 16 and l < lim_x and col_mean(l) < th:
        l += step
    while r - step > l + 16 and w - 1 - r < lim_x and col_mean(r) < th:
        r -= step

    if (b - t + 1) * (r - l + 1) < h * w * 0.45:
        return bgr, False
    if t == 0 and b == h - 1 and l == 0 and r == w - 1:
        return bgr, False
    return bgr[t : b + 1, l : r + 1].copy(), True

def _deskew_projection_score(gray: np.ndarray, angle_deg: float) -> float:
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rot = cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    proj = np.sum(rot.astype(np.float64), axis=1)
    d = np.diff(proj)
    return float(np.sum(d * d))

def _deskew_sweep_coarse(
    work: np.ndarray, lo: float, hi: float, step: float
) -> tuple[float, float, float]:
    s0 = _deskew_projection_score(work, 0.0)
    best_a = 0.0
    best_s = -1.0
    a = lo
    while a <= hi + 1e-6:
        s = _deskew_projection_score(work, float(a))
        if s > best_s:
            best_s = s
            best_a = float(a)
        a += step
    return best_a, best_s, s0

def _deskew_refine(
    work: np.ndarray, best_a: float, best_s: float, lo: float, hi: float, cfg: PreprocessConfig
) -> tuple[float, float]:
    r0 = max(cfg.deskew_refine_radius_deg, cfg.deskew_step_deg)
    lo2 = max(lo, best_a - r0)
    hi2 = min(hi, best_a + r0)
    rs = cfg.deskew_refine_step_deg
    a = lo2
    cur_a, cur_s = best_a, best_s
    while a <= hi2 + 1e-9:
        s = _deskew_projection_score(work, float(a))
        if s > cur_s:
            cur_s = s
            cur_a = float(a)
        a += rs
    return cur_a, cur_s

def _estimate_deskew_deg(gray: np.ndarray, cfg: PreprocessConfig) -> float:
    work_gray = _center_crop_gray(gray, cfg.deskew_use_center_crop_frac)
    blur = cv2.GaussianBlur(work_gray, (3, 3), 0)
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    work = bin_img

    lo, hi = cfg.deskew_angle_range_deg
    step = cfg.deskew_step_deg
    best_a, best_s, s0 = _deskew_sweep_coarse(work, lo, hi, step)
    best_a, best_s = _deskew_refine(work, best_a, best_s, lo, hi, cfg)

    weak = abs(best_a) < 1.0 and best_s <= s0 * cfg.deskew_weak_score_ratio_max
    if weak:
        elo, ehi = cfg.deskew_extend_angle_range_deg
        estep = cfg.deskew_weak_extend_step_deg
        best_a, best_s, _ = _deskew_sweep_coarse(work, elo, ehi, estep)
        best_a, best_s = _deskew_refine(work, best_a, best_s, elo, ehi, cfg)

    s0f = _deskew_projection_score(work, 0.0)
    if abs(best_a) < 0.4 and best_s <= s0f * 1.02:
        best_a = 0.0

    return best_a

def _top_bottom_row_var_ratio(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    gray = gray[int(0.05 * h) : int(0.95 * h), int(0.05 * w) : int(0.95 * w)]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hh = b.shape[0]

    def band_var(y0: int, y1: int) -> float:
        block = b[y0:y1].astype(np.float64)
        return float(np.var(np.sum(block, axis=1))) + 1e-6

    t = band_var(0, max(4, hh // 6))
    bo = band_var(hh - max(4, hh // 6), hh)
    return float(t / bo)

def _rotate_k_quarters_ccw(bgr: np.ndarray, k: int) -> np.ndarray:
    x = bgr
    for _ in range(int(k) % 4):
        x = cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return x

def _is_portrait_bgr(bgr: np.ndarray, h_over_w_min: float) -> bool:
    h, w = bgr.shape[:2]
    return h > w * h_over_w_min

def _pick_post_deskew_quarter_turn(
    bgr: np.ndarray, cfg: PreprocessConfig
) -> tuple[np.ndarray, int, float, dict[str, Any]]:
    ratio_deskewed = _top_bottom_row_var_ratio(bgr)
    meta: dict[str, Any] = {
        "post_deskew_tb_ratio_deskewed": ratio_deskewed,
        "post_deskew_upside_down_signal": ratio_deskewed > cfg.upright_180_ratio_threshold,
        "post_deskew_ambiguous_0_180": False,
        "post_deskew_180_resolution": "none",
    }
    if not cfg.auto_upright_180:
        return bgr, 0, ratio_deskewed, meta

    ph = cfg.post_deskew_portrait_h_over_w_min
    allowed = [k for k in range(4) if _is_portrait_bgr(_rotate_k_quarters_ccw(bgr, k), ph)]
    if not allowed:
        allowed = [0, 1, 2, 3]

    entries: list[tuple[int, float, float]] = []
    for k in allowed:
        cand = _rotate_k_quarters_ccw(bgr, k)
        entries.append((k, _horizontal_line_energy(cand), _top_bottom_row_var_ratio(cand)))

    max_e = max(e for _, e, _ in entries)
    floor = max_e * cfg.post_deskew_quarter_energy_floor_ratio
    near = [(k, e, r) for k, e, r in entries if e >= floor]
    near_ks = {t[0] for t in near}

    if 0 in near_ks and 2 in near_ks:
        meta["post_deskew_ambiguous_0_180"] = True
        e0 = next(e for k, e, _ in entries if k == 0)
        e2 = next(e for k, e, _ in entries if k == 2)
        margin = cfg.post_deskew_180_energy_prefer_k2_margin
        if ratio_deskewed > cfg.upright_180_ratio_threshold:
            near = [t for t in near if t[0] != 0]
            meta["post_deskew_180_resolution"] = "forced_k2_tb_high"
        elif e2 > e0 * (1.0 + margin):
            meta["post_deskew_180_resolution"] = "prefer_k2_energy"
        else:
            near = [t for t in near if t[0] != 2]
            meta["post_deskew_180_resolution"] = "prefer_k0_tb_ok"

    near.sort(key=lambda t: (t[2], -t[1], t[0]))
    best_k, _, ratio_out = near[0]
    return _rotate_k_quarters_ccw(bgr, best_k), best_k, ratio_out, meta

def _order_quad_pts(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def _try_document_warp(bgr: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, bool]:
    h0, w0 = bgr.shape[:2]
    in_area = float(h0 * w0)
    scale = cfg.warp_max_side_for_search / max(h0, w0)
    if scale >= 1.0:
        small = bgr
        inv_scale = 1.0
    else:
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        inv_scale = 1.0 / scale

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sh, sw = small.shape[:2]
    min_area = (sw * sh) * cfg.contour_min_area_ratio
    best = None
    best_area = 0.0

    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if a > best_area:
            best_area = a
            best = approx.reshape(4, 2).astype(np.float32)

    if best is None:
        return bgr, False

    if best_area > (sh * sw) * cfg.warp_max_area_ratio:
        return bgr, False

    best *= inv_scale
    rect = _order_quad_pts(best)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width = max(width, 1)
    height = max(height, 1)

    ar = max(width, height) / max(min(width, height), 1)
    if ar > cfg.warp_max_aspect_ratio:
        return bgr, False

    out_area = float(width * height)
    if out_area < in_area * cfg.warp_min_output_area_ratio:
        return bgr, False

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(bgr, m, (width, height), flags=cv2.INTER_LINEAR)
    return warped, True

def _bbox_crop_largest_region(bgr: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, bool]:
    h0, w0 = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, False
    min_area = h0 * w0 * cfg.bbox_crop_min_area_ratio
    best = None
    best_area = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        if a > best_area:
            best_area = a
            best = c
    if best is None:
        return bgr, False
    x, y, w, h = cv2.boundingRect(best)
    pad_x = max(2, int(w * cfg.bbox_crop_pad_ratio))
    pad_y = max(2, int(h * cfg.bbox_crop_pad_ratio))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w0, x + w + pad_x)
    y1 = min(h0, y + h + pad_y)
    if (x1 - x0) * (y1 - y0) < h0 * w0 * 0.28:
        return bgr, False
    return bgr[y0:y1, x0:x1].copy(), True

def _gray_clahe_denoise(bgr: np.ndarray, cfg: PreprocessConfig, denoise: bool) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=cfg.denoise_h,
            templateWindowSize=cfg.denoise_template_window,
            searchWindowSize=cfg.denoise_search_window,
        )
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid_size)
    return clahe.apply(gray)

def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _adaptive_binary(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    bs = cfg.adaptive_block_size
    if bs % 2 == 0:
        bs += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs,
        cfg.adaptive_c,
    )

def preprocess(
    image_bgr: np.ndarray,
    profile: Profile = "default",
    config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PreprocessConfig()
    steps: dict[str, np.ndarray] = {}
    steps["01_raw"] = image_bgr.copy()

    img = _resize_max_side(image_bgr, cfg.max_side)
    steps["02_resized"] = img.copy()

    orient_k = 0
    orient_flip = 0
    orient_e = 0.0
    orient_skipped = False
    if profile != "minimal":
        img, orient_k, orient_flip, orient_e, orient_skipped = _pick_page_orientation(img, cfg)
    steps["02b_orient"] = img.copy()

    force180 = False
    if cfg.force_rotate_180 and profile != "minimal":
        img = cv2.rotate(img, cv2.ROTATE_180)
        force180 = True
        steps["02c_force180"] = img.copy()

    warped = False
    if cfg.try_perspective_warp and profile != "minimal":
        img, warped = _try_document_warp(img, cfg)
    steps["03_warped" if warped else "03_no_warp"] = img.copy()

    bbox_cropped = False
    if (not warped) and cfg.fallback_bbox_crop and profile != "minimal":
        img, bbox_cropped = _bbox_crop_largest_region(img, cfg)
    if bbox_cropped:
        steps["03c_bbox_crop"] = img.copy()

    trimmed = False
    if profile != "minimal":
        img, trimmed = _trim_uniform_edges(img, cfg)
    if trimmed:
        steps["03d_trim"] = img.copy()

    deskew_deg = 0.0
    post_deskew_quarter_ccw = 0
    upright_ratio = 0.0
    post_deskew_meta: dict[str, Any] = {}
    denoise = profile != "minimal"
    if cfg.auto_deskew and profile != "minimal":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        deskew_deg = _estimate_deskew_deg(g, cfg)
    total_rot = deskew_deg + cfg.rotate_deg
    img = _rotate(img, total_rot)
    steps["04a_deskewed"] = img.copy()
    if profile != "minimal":
        img, post_deskew_quarter_ccw, upright_ratio, post_deskew_meta = _pick_post_deskew_quarter_turn(img, cfg)
    else:
        post_deskew_meta = {
            "post_deskew_tb_ratio_deskewed": 0.0,
            "post_deskew_upside_down_signal": False,
            "post_deskew_ambiguous_0_180": False,
            "post_deskew_180_resolution": "profile_minimal",
        }
    steps["04b_straightened"] = img.copy()

    clahe_gray = _gray_clahe_denoise(img, cfg, denoise=denoise)
    steps["05_clahe_gray"] = clahe_gray.copy()

    out_main = _gray_to_bgr(clahe_gray)
    steps["06_for_ocr_default"] = out_main.copy()

    aggressive_extra: np.ndarray | None = None
    if profile == "aggressive":
        bin_gray = _adaptive_binary(clahe_gray, cfg)
        aggressive_extra = _gray_to_bgr(bin_gray)
        steps["07_for_ocr_adaptive_bgr"] = aggressive_extra.copy()

    result: dict[str, Any] = {
        "image_for_ocr": out_main,
        "image_for_ocr_alt": aggressive_extra,
        "warp_applied": warped,
        "bbox_crop_applied": bbox_cropped,
        "border_trim_applied": trimmed,
        "page_orient_quarter_ccw": orient_k,
        "page_orient_flip_180": orient_flip,
        "page_orient_energy": orient_e,
        "page_orient_skipped": orient_skipped,
        "force_rotate_180_applied": force180,
        "deskew_deg": deskew_deg,
        "rotation_applied_deg": total_rot,
        "post_deskew_quarter_ccw": post_deskew_quarter_ccw,
        "upright_180_applied": post_deskew_quarter_ccw == 2,
        "upright_180_row_var_ratio": upright_ratio,
        "profile": profile,
        "steps": steps,
        **post_deskew_meta,
    }
    return result

def save_debug_steps(steps: dict[str, np.ndarray], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, arr in sorted(steps.items()):
        p = out / f"{name}.png"
        cv2.imwrite(str(p), arr)
        saved.append(p)
    return saved
