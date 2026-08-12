"""Tests for the checked-in hand-labeled OOD dataset."""

from pathlib import Path

from research.data.ood_io import load_ood_jsonl
from research.data.ood_stats import validate_complete_ood

DATASET = Path("research/labeled_ood/examples.jsonl")


def test_ood_dataset_loads() -> None:
    """The checked-in OOD dataset should load without errors."""
    examples = load_ood_jsonl(DATASET)

    assert examples


def test_ood_dataset_ids_are_unique() -> None:
    """Every hand-labeled example needs a unique stable ID."""
    examples = load_ood_jsonl(DATASET)
    ids = [example.example_id for example in examples]

    assert len(ids) == len(set(ids))


def test_ood_dataset_contains_negative_examples() -> None:
    """The benchmark should include examples containing no PII."""
    examples = load_ood_jsonl(DATASET)

    assert any(not example.spans for example in examples)


def test_ood_dataset_contains_pii() -> None:
    """The benchmark should include positively labeled PII."""
    examples = load_ood_jsonl(DATASET)

    assert any(example.spans for example in examples)


def test_ood_dataset_is_complete() -> None:
    """The Stage 5 benchmark must contain exactly 200 examples."""
    validate_complete_ood(load_ood_jsonl(DATASET))
