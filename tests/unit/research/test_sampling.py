"""Tests for deterministic dataset sampling."""

import pytest

from pii_scrub.text import CharacterSpan
from research.data.models import DatasetExample
from research.data.splits import sample_examples


def _make_examples(
    count: int,
) -> tuple[DatasetExample, ...]:
    """Create small normalized examples for splitting tests."""

    return tuple(
        DatasetExample(
            example_id=f"example-{index}",
            text=f"User {index}",
            spans=(
                CharacterSpan(
                    start=0,
                    end=4,
                    entity_type="PERSON",
                ),
            ),
            source="test",
            language="en",
        )
        for index in range(count)
    )


def _example_ids(
    examples: tuple[DatasetExample, ...],
) -> tuple[str, ...]:
    """Return example IDs from a tuple."""

    return tuple(example.example_id for example in examples)


def test_sample_examples_returns_requested_size() -> None:
    """Sampling should return exactly the requested number of rows."""

    examples = _make_examples(10)

    result = sample_examples(
        examples,
        sample_size=4,
        seed=42,
    )

    assert len(result) == 4


def test_sample_examples_is_deterministic() -> None:
    """The same seed should produce the same sample."""

    examples = _make_examples(20)

    first = sample_examples(
        examples,
        sample_size=5,
        seed=42,
    )
    second = sample_examples(
        examples,
        sample_size=5,
        seed=42,
    )

    assert _example_ids(first) == _example_ids(second)


def test_sample_examples_changes_with_seed() -> None:
    """Different seeds should normally produce different samples."""

    examples = _make_examples(20)

    first = sample_examples(
        examples,
        sample_size=5,
        seed=42,
    )
    second = sample_examples(
        examples,
        sample_size=5,
        seed=99,
    )

    assert _example_ids(first) != _example_ids(second)


def test_sample_examples_does_not_modify_input() -> None:
    """Sampling must preserve the original example order."""

    examples = _make_examples(10)
    original_ids = _example_ids(examples)

    sample_examples(
        examples,
        sample_size=5,
        seed=42,
    )

    assert _example_ids(examples) == original_ids


def test_sample_examples_rejects_oversized_sample() -> None:
    """A sample cannot contain more rows than the dataset."""

    with pytest.raises(
        ValueError,
        match="exceeds dataset size",
    ):
        sample_examples(
            _make_examples(3),
            sample_size=4,
        )


def test_sample_examples_rejects_non_integer_seed() -> None:
    """A deterministic seed must be an integer."""

    with pytest.raises(
        TypeError,
        match="seed must be an integer",
    ):
        sample_examples(
            _make_examples(5),
            sample_size=2,
            seed="42",  # type: ignore[arg-type]
        )
