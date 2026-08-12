"""Tests for OOD benchmark summaries and validation."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.data.ood_stats import (
    summarize_ood,
    validate_complete_ood,
    validate_ood,
)


def _example(
    example_id: str,
    *spans: CharacterSpan,
) -> DatasetExample:
    """Return one small OOD example for summary tests.

    Example:
        ``_example("ood-1")`` creates a negative example.
    """
    return DatasetExample(
        example_id=example_id,
        text="John email",
        spans=spans,
        source="hand_labeled_ood",
        language="en",
    )


def test_summarizes_ood_dataset() -> None:
    """Summary counts should reflect examples and entity labels."""
    examples = [
        _example("ood-1", CharacterSpan(0, 4, "PERSON")),
        _example("ood-2"),
        _example("ood-3", CharacterSpan(5, 10, "EMAIL")),
    ]

    assert summarize_ood(examples) == {
        "examples": 3,
        "positive": 2,
        "negative": 1,
        "entities": 2,
        "labels": {"EMAIL": 1, "PERSON": 1},
    }


def test_rejects_duplicate_ids() -> None:
    """Duplicate benchmark IDs should fail validation."""
    examples = [_example("ood-1"), _example("ood-1")]

    with pytest.raises(ValueError, match="IDs must be unique"):
        validate_ood(examples)


def test_rejects_more_than_target() -> None:
    """The benchmark should not silently exceed its planned size."""
    examples = [_example(f"ood-{index}") for index in range(3)]

    with pytest.raises(ValueError, match="exceeds target"):
        validate_ood(examples, target=2)


def test_allows_dataset_under_construction() -> None:
    """A partial benchmark should remain valid while labeling."""
    validate_ood([_example("ood-1")], target=200)


def test_complete_ood_requires_target() -> None:
    """Final validation should reject an incomplete benchmark."""
    examples = [_example(f"ood-{index}") for index in range(199)]

    with pytest.raises(ValueError, match="exactly 200"):
        validate_complete_ood(examples)


def test_complete_ood_accepts_target() -> None:
    """Final validation should accept exactly 200 examples."""
    examples = [_example(f"ood-{index}") for index in range(200)]

    validate_complete_ood(examples)
