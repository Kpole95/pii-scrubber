"""Tests for Qwen causal-LM batch collation."""

import pytest
import torch

from research.train.qwen_collator import collate_qwen_records
from research.train.qwen_dataset import QwenTrainingRecord
from research.train.qwen_tokens import IGNORE_INDEX


def _record(
    input_ids: list[int],
    labels: list[int],
) -> QwenTrainingRecord:
    """Build one compact test record."""
    return QwenTrainingRecord(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
    )


def test_pads_variable_length_records() -> None:
    """Short records should be right-padded to the longest batch member."""

    batch = collate_qwen_records(
        [
            _record(
                [10, 11, 12],
                [IGNORE_INDEX, 11, 12],
            ),
            _record(
                [20, 21],
                [IGNORE_INDEX, 21],
            ),
        ],
        pad_token_id=99,
    )

    assert torch.equal(
        batch["input_ids"],
        torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 99],
            ]
        ),
    )

    assert torch.equal(
        batch["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ]
        ),
    )


def test_padding_is_ignored_by_loss() -> None:
    """Padding positions should use the causal-LM ignore index."""

    batch = collate_qwen_records(
        [
            _record(
                [10, 11, 12],
                [IGNORE_INDEX, 11, 12],
            ),
            _record(
                [20],
                [20],
            ),
        ],
        pad_token_id=99,
    )

    assert torch.equal(
        batch["labels"],
        torch.tensor(
            [
                [IGNORE_INDEX, 11, 12],
                [20, IGNORE_INDEX, IGNORE_INDEX],
            ]
        ),
    )


def test_preserves_existing_prompt_mask() -> None:
    """Prompt positions already masked with -100 must remain masked."""

    batch = collate_qwen_records(
        [
            _record(
                [10, 11, 12],
                [IGNORE_INDEX, IGNORE_INDEX, 12],
            )
        ],
        pad_token_id=99,
    )

    assert batch["labels"].tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 12]]


def test_returns_long_tensors() -> None:
    """Causal-LM model inputs should use integer tensors."""

    batch = collate_qwen_records(
        [_record([10], [10])],
        pad_token_id=99,
    )

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_rejects_empty_batch() -> None:
    """Trainer should never collate an empty feature list."""

    with pytest.raises(ValueError, match="must not be empty"):
        collate_qwen_records(
            [],
            pad_token_id=99,
        )


def test_rejects_negative_pad_token() -> None:
    """Padding IDs must be valid vocabulary indices."""

    with pytest.raises(ValueError, match="non-negative"):
        collate_qwen_records(
            [_record([10], [10])],
            pad_token_id=-1,
        )


def test_rejects_misaligned_attention_mask() -> None:
    """Every training array must describe the same sequence length."""

    record = QwenTrainingRecord(
        input_ids=[10, 11],
        attention_mask=[1],
        labels=[10, 11],
    )

    with pytest.raises(ValueError, match="attention_mask"):
        collate_qwen_records(
            [record],
            pad_token_id=99,
        )


def test_rejects_misaligned_labels() -> None:
    """Labels must align one-for-one with model input tokens."""

    record = QwenTrainingRecord(
        input_ids=[10, 11],
        attention_mask=[1, 1],
        labels=[10],
    )

    with pytest.raises(ValueError, match="labels length"):
        collate_qwen_records(
            [record],
            pad_token_id=99,
        )


def test_rejects_empty_record() -> None:
    """Individual training records must contain model tokens."""

    record = QwenTrainingRecord(
        input_ids=[],
        attention_mask=[],
        labels=[],
    )

    with pytest.raises(ValueError, match="training records must not be empty"):
        collate_qwen_records(
            [record],
            pad_token_id=99,
        )
