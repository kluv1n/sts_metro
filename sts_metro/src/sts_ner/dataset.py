
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sts_ner.bio import align_entity_spans, build_label_list

@dataclass
class StsNerDataset:
    samples: list[dict[str, Any]]
    tokenizer: Any
    label2id: dict[str, int]
    max_length: int = 512

    def __len__(self) -> int:
        return len(self.samples)

    def _token_label(self, char_tags: list[str], ts: int, te: int) -> int:
        if ts is None or te is None or (ts == 0 and te == 0):
            return -100
        if ts >= te:
            return self.label2id["O"]
        first = None
        for i in range(ts, min(te, len(char_tags))):
            if char_tags[i] != "O":
                first = i
                break
        if first is None:
            return self.label2id["O"]
        tag = char_tags[first]
        if tag.startswith("B-"):
            if first == ts:
                return self.label2id[tag]
            et = tag[2:]
            return self.label2id.get(f"I-{et}", self.label2id["O"])
        if tag.startswith("I-"):
            return self.label2id[tag]
        return self.label2id["O"]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        text = row["text"]
        entities = row.get("entities") or {}

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding=False,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        char_len = len(text)
        char_tags = ["O"] * char_len
        for s, e, et in align_entity_spans(text, entities):
            if 0 <= s < e <= char_len:
                char_tags[s] = f"B-{et}"
                for p in range(s + 1, e):
                    char_tags[p] = f"I-{et}"

        labels: list[int] = []
        for (ts, te) in offsets:
            labels.append(self._token_label(char_tags, ts, te))

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

def collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    label_pad: int = -100,
) -> dict[str, Any]:
    import torch

    max_len = max(len(x["input_ids"]) for x in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((bs, max_len), dtype=torch.long)
    labels = torch.full((bs, max_len), label_pad, dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        attn[i, :n] = torch.tensor(x["attention_mask"], dtype=torch.long)
        labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
