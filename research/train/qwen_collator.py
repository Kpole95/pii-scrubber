"""Batch variable-length Qwen training records for causal-LM training."""

from collections.abc import Sequence

import torch
from torch import Tensor

from research.train.qwen_dataset import QwenTrainingRecord
from research.train.qwen_tokens import IGNORE_INDEX


def collate_qwen_records(
    records: Sequence[QwenTrainingRecord],
    *,
    pad_token_id: int,
) -> dict[str, Tensor]:
    """Right-pad Qwen records while preserving assistant-only loss labels."""

    if not records:
        raise ValueError("records must not be empty")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")

    max_length = max(len(record["input_ids"]) for record in records)

    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []

    for record in records:
        _validate_record(record)

        padding = max_length - len(record["input_ids"])

        input_ids.append(record["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(record["attention_mask"] + [0] * padding)
        labels.append(record["labels"] + [IGNORE_INDEX] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _validate_record(record: QwenTrainingRecord) -> None:
    """Require aligned non-empty causal-LM training arrays."""

    input_length = len(record["input_ids"])

    if input_length == 0:
        raise ValueError("training records must not be empty")

    if len(record["attention_mask"]) != input_length:
        raise ValueError("attention_mask length must match input_ids")

    if len(record["labels"]) != input_length:
        raise ValueError("labels length must match input_ids")
