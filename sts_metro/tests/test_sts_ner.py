
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sts_ner.bio import align_entity_spans, build_label_list, find_entity_span_simple
from sts_ner.dataset import StsNerDataset

def test_find_span_simple():
    text = "СОБСТВЕННИК\nИВАНОВ\nIVANOV\nСЕРГЕЙ"
    i = text.find("ИВАНОВ")
    assert find_entity_span_simple(text, "ИВАНОВ") == (i, i + len("ИВАНОВ"))

def test_align_non_overlapping():
    text = "OWNER\nИВАНОВ\nIVANOV\nСЕРГЕЙ\nSERGEI\nПЕТРОВИЧ"
    ent = {
        "surname_ru": "ИВАНОВ",
        "name_ru": "СЕРГЕЙ",
        "patronymic_ru": "ПЕТРОВИЧ",
        "surname_en": "IVANOV",
        "name_en": "SERGEI",
    }
    spans = align_entity_spans(text, ent)
    assert len(spans) == 5

@pytest.mark.skipif(
    not Path(ROOT / "source_model_train" / "splits" / "train.json").is_file(),
    reason="run scripts/prepare_ner_splits.py first",
)
def test_dataset_tokenizes():
    import os

    hf_home = ROOT / ".hf_test_cache"
    hf_home.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))

    transformers = pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    path = ROOT / "source_model_train" / "splits" / "train.json"
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"][:3]
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    label2id = {l: i for i, l in enumerate(build_label_list())}
    ds = StsNerDataset(samples, tok, label2id, 256)
    row = ds[0]
    assert len(row["input_ids"]) == len(row["labels"])
    assert all(x >= -100 for x in row["labels"])

def test_collate_runs():
    import os

    hf_home = ROOT / ".hf_test_cache"
    hf_home.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from sts_ner.dataset import collate_fn

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
    label2id = {l: i for i, l in enumerate(build_label_list())}
    text = "СОБСТВЕННИК\nИВАНОВ\nIVANOV\nСЕРГЕЙ\nSERGEI\nПЕТРОВИЧ"
    ent = {
        "surname_ru": "ИВАНОВ",
        "name_ru": "СЕРГЕЙ",
        "patronymic_ru": "ПЕТРОВИЧ",
        "surname_en": "IVANOV",
        "name_en": "SERGEI",
    }
    ds = StsNerDataset([{"text": text, "entities": ent}], tok, label2id, 256)
    pad = tok.pad_token_id or 0
    batch = collate_fn([ds[0], ds[0]], pad_token_id=int(pad))
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape
