
from __future__ import annotations

import re
from typing import Any

ENTITY_TYPES = (
    "SURNAME_RU",
    "NAME_RU",
    "PATRONYMIC_RU",
    "SURNAME_EN",
    "NAME_EN",
)
ENTITY_KEYS = (
    "surname_ru",
    "name_ru",
    "patronymic_ru",
    "surname_en",
    "name_en",
)

def build_label_list() -> list[str]:
    labels = ["O"]
    for et in ENTITY_TYPES:
        labels.append(f"B-{et}")
        labels.append(f"I-{et}")
    return labels

def find_entity_span_simple(text: str, value: str) -> tuple[int, int] | None:
    v = (value or "").strip()
    if not v:
        return None
    i = text.find(v)
    if i >= 0:
        return i, i + len(v)
    i = text.lower().find(v.lower())
    if i >= 0:
        return i, i + len(v)
    
    v2 = re.sub(r"\s+", "", v)
    if len(v2) >= 2:
        t2 = re.sub(r"\s+", "", text)
        j = t2.lower().find(v2.lower())
        if j >= 0:
            
            flat = 0
            start_o = None
            end_o = None
            for idx, ch in enumerate(text):
                if ch.isspace():
                    continue
                if flat == j:
                    start_o = idx
                if flat == j + len(v2) - 1:
                    end_o = idx + 1
                    break
                flat += 1
            if start_o is not None and end_o is not None:
                return start_o, end_o
    return None

def align_entity_spans(text: str, entities: dict[str, str]) -> list[tuple[int, int, str]]:
    taken: list[tuple[int, int]] = []
    out: list[tuple[int, int, str]] = []

    for key, et in zip(ENTITY_KEYS, ENTITY_TYPES):
        val = entities.get(key) or ""
        sp = find_entity_span_simple(text, val)
        if sp is None:
            continue
        s, e = sp
        if e <= s:
            continue
        overlap = any(not (e <= ts or s >= te) for ts, te in taken)
        if overlap:
            continue
        taken.append((s, e))
        out.append((s, e, et))
    return out

def labels_to_entity_strings(
    text: str,
    offset_mapping: list[tuple[int, int]],
    pred_ids: list[int],
    id2label: dict[int, str],
) -> dict[str, str]:
    chunks: dict[str, list[str]] = {k: [] for k in ENTITY_KEYS}
    current_key: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current_key, buf
        if current_key and buf:
            st = "".join(buf).strip()
            if st:
                chunks[current_key].append(st)
        buf = []
        current_key = None

    key_by_type = dict(zip(ENTITY_TYPES, ENTITY_KEYS))

    for (s, e), tid in zip(offset_mapping, pred_ids):
        if s is None or e is None or (s == 0 and e == 0):
            continue
        lab = id2label.get(int(tid), "O")
        if lab == "O":
            flush()
            continue
        parts = lab.split("-", 1)
        if len(parts) != 2:
            continue
        pref, et = parts[0], parts[1]
        key = key_by_type.get(et)
        if not key:
            continue
        piece = text[s:e]
        if pref == "B":
            flush()
            current_key = key
            buf.append(piece)
        elif pref == "I" and current_key == key:
            buf.append(piece)
        else:
            flush()
            current_key = key
            buf.append(piece)
    flush()
    return {k: " ".join(v).strip() for k, v in chunks.items()}
