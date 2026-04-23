
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_MAX_ANCHOR_LINE = 22

_ANCHOR = re.compile(
    r"собственн|владельц|владелец|вхаделец|в\^аделец",
    re.IGNORECASE,
)
_STOP_LINE = re.compile(
    r"^(респуб|респу|москв|район|нас\.?\s*п|насел|пункт|ул(ица|\.)|ухиц|"
    r"особые|осооые|осоо|дом\b|корп|кв\.?|кварт|област|край)",
    re.IGNORECASE,
)

_STOP_SUBSTR = re.compile(
    r"федерац|особые\s*отмет|осооые\s*отмет",
    re.IGNORECASE,
)
_LATIN_WORD = re.compile(r"^[A-Za-z\-]{3,}$")

_MIXED_SCRIPT = re.compile(r"[A-Za-z][А-Яа-яёЁ]|[А-Яа-яёЁ][A-Za-z]")
_CY_TOKEN = re.compile(r"[А-Яа-яЁё\-]{2,}")
_LAT_TOKEN = re.compile(r"[A-Za-z\-]{2,}")

def _cyrillic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    cy = sum(1 for c in letters if "\u0400" <= c <= "\u04ff" or c in "ёЁ")
    return cy / len(letters)

def _clean_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _strip_leading_parenthetical(s: str) -> str:
    return re.sub(r"^\([^)]{0,40}\)\s*", "", s.strip())

def _is_cyrillic_name_candidate(line: str) -> bool:
    line = _clean_line(line)
    if len(line) < 2:
        return False
    if _MIXED_SCRIPT.search(line):
        return False
    if re.search(r"\d{4,}", line):
        return False
    if _LATIN_WORD.match(line.replace(" ", "")) and _cyrillic_ratio(line) < 0.15:
        return False
    if _cyrillic_ratio(line) < 0.75:
        return False
    if len(line) > 80:
        return False
    return True

def _is_latin_name_candidate(line: str) -> bool:
    line = _clean_line(line)
    if len(line) < 2 or len(line) > 80:
        return False
    if _MIXED_SCRIPT.search(line):
        return False
    if re.search(r"\d{3,}", line):
        return False
    if _cyrillic_ratio(line) > 0.08:
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 2:
        return False
    if not re.search(r"[A-Za-z]", line):
        return False
    return True

def _anchor_line_index(lines: list[str]) -> int | None:
    for i, raw in enumerate(lines[:_MAX_ANCHOR_LINE]):
        t = raw.strip()
        if not t:
            continue
        if _ANCHOR.search(t):
            return i
    return None

def _starts_with_stop(line: str) -> bool:
    t = _clean_line(line)
    if not t:
        return False
    if _STOP_SUBSTR.search(t):
        return True
    return bool(_STOP_LINE.match(t))

def _cyrillic_confidence(parts: list[str]) -> tuple[Literal["high", "medium", "low"], str]:
    if not parts:
        return "low", "empty"
    note = "ok"
    conf: Literal["high", "medium", "low"] = "medium"
    if len(parts) < 3:
        conf = "low"
        note = "fewer_than_three_parts"
    last = parts[-1].lower()
    if len(parts) == 3 and not re.search(r"(вна|ич|оглы|кызы)$", last):
        conf = "medium"
        note = "patronymic_suffix_uncertain"
    if len(parts) == 3 and all(_cyrillic_ratio(p) > 0.55 for p in parts):
        conf = "high"
        note = "ok"
    return conf, note

def _latin_confidence(parts: list[str]) -> tuple[Literal["high", "medium", "low"], str]:
    if not parts:
        return "low", "empty"
    if len(parts) >= 3:
        return "high", "ok"
    if len(parts) == 2:
        return "medium", "two_parts_only"
    return "low", "fewer_than_three_parts"

def _collect_name_parts(lines: list[str], idx: int, cyrillic: bool) -> list[str]:
    parts: list[str] = []
    pred = _is_cyrillic_name_candidate if cyrillic else _is_latin_name_candidate
    for j in range(idx + 1, len(lines)):
        raw = lines[j]
        line = _clean_line(raw)
        if not line:
            continue
        if _starts_with_stop(line):
            break
        if _ANCHOR.search(line) and not re.search(r"[А-ЯЁ]{2,}", line):
            continue
        if not pred(line):
            continue
        line = _strip_leading_parenthetical(line)
        if not pred(line):
            continue
        parts.append(line)
        if len(parts) >= 3:
            break
    return parts

def _pick_primary(
    parts_cy: list[str],
    conf_cy: Literal["high", "medium", "low"],
    parts_lat: list[str],
    conf_lat: Literal["high", "medium", "low"],
) -> tuple[list[str], Literal["high", "medium", "low"], str, str]:
    lat_full = len(parts_lat) == 3 and conf_lat == "high"
    cy_usable = len(parts_cy) == 3 and conf_cy in ("high", "medium")

    if lat_full and (len(parts_cy) < 3 or conf_cy == "low"):
        return parts_lat, conf_lat, "en", "latin_full_russian_weak"
    if cy_usable:
        return parts_cy, conf_cy, "ru", "cyrillic_triple_ok"
    if len(parts_cy) == 3:
        return parts_cy, conf_cy, "ru", "cyrillic_triple_low_conf"
    if lat_full:
        return parts_lat, conf_lat, "en", "latin_full"
    if len(parts_cy) >= len(parts_lat) and parts_cy:
        return parts_cy, conf_cy, "ru", "cyrillic_by_coverage"
    if parts_lat:
        return parts_lat, conf_lat, "en", "latin_fallback"
    return parts_cy, conf_cy, ("none" if not parts_cy else "ru"), "empty_or_cyrillic_only"

def _pick_name_triplet(parts: list[str], cyrillic: bool) -> list[str]:
    token_re = _CY_TOKEN if cyrillic else _LAT_TOKEN
    out: list[str] = []
    for ln in parts:
        for tok in token_re.findall(ln):
            tok = tok.strip("-")
            if len(tok) < 2:
                continue
            out.append(tok)
            if len(out) >= 3:
                return out
    return out

def _split_5_entities(parts_cy: list[str], parts_lat: list[str]) -> dict[str, str]:
    triplet_cy = _pick_name_triplet(parts_cy, cyrillic=True)
    triplet_lat = _pick_name_triplet(parts_lat, cyrillic=False)
    return {
        "surname_ru": triplet_cy[0] if len(triplet_cy) >= 1 else "",
        "name_ru": triplet_cy[1] if len(triplet_cy) >= 2 else "",
        "patronymic_ru": triplet_cy[2] if len(triplet_cy) >= 3 else "",
        "surname_en": triplet_lat[0] if len(triplet_lat) >= 1 else "",
        "name_en": triplet_lat[1] if len(triplet_lat) >= 2 else "",
    }

def extract_fio_from_ocr_text(text: str) -> dict[str, object]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]

    idx = _anchor_line_index(lines)
    parts_cy: list[str] = []
    parts_lat: list[str] = []
    entities_5 = _split_5_entities(parts_cy, parts_lat)
    if idx is None:
        return {
            "fio": "",
            "fio_cyrillic": "",
            "fio_latin": "",
            "parts_cyrillic": [],
            "parts_latin": [],
            "parts": [],
            "confidence": "low",
            "confidence_cyrillic": "low",
            "confidence_latin": "low",
            "primary_language": "none",
            "primary_reason": "no_owner_anchor",
            "note": "no_owner_anchor",
            **entities_5,
        }

    parts_cy = _collect_name_parts(lines, idx, cyrillic=True)
    parts_lat = _collect_name_parts(lines, idx, cyrillic=False)
    entities_5 = _split_5_entities(parts_cy, parts_lat)

    fio_cy = " ".join(parts_cy)
    fio_lat = " ".join(parts_lat)

    conf_cy, note_cy = _cyrillic_confidence(parts_cy)
    conf_lat, note_lat = _latin_confidence(parts_lat)

    parts, conf, primary_lang, reason = _pick_primary(parts_cy, conf_cy, parts_lat, conf_lat)
    fio = " ".join(parts)
    if not parts_cy and not parts_lat:
        note = "no_name_lines_after_anchor"
        reason = "no_name_lines_after_anchor"
    else:
        note = note_cy if primary_lang == "ru" else note_lat
        if primary_lang == "en" and parts_lat and parts_cy:
            note = f"{reason};{note_lat}"

    return {
        "fio": fio,
        "fio_cyrillic": fio_cy,
        "fio_latin": fio_lat,
        "parts_cyrillic": parts_cy,
        "parts_latin": parts_lat,
        "parts": parts,
        "confidence": conf,
        "confidence_cyrillic": conf_cy,
        "confidence_latin": conf_lat,
        "primary_language": primary_lang,
        "primary_reason": reason,
        "note": note,
        **entities_5,
    }

__all__ = ["extract_fio_from_ocr_text"]
