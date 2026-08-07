"""Tests for deterministic dataset sampling and splitting."""

import pytest

from pii_scrub.text import CharacterSpan
from research.data.models import DatasetExample
from research.data.splits import (
    DatasetSplit,
    sample_examples,
    split_examples,
)


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


def test_split_examples_returns_expected_counts() -> None:
    """Requested validation and test sizes should be respected."""

    result = split_examples(
        _make_examples(10),
        validation_size=2,
        test_size=3,
        seed=42,
    )

    assert result.train_count == 5
    assert result.validation_count == 2
    assert result.test_count == 3
    assert result.total_count == 10


def test_split_examples_is_deterministic() -> None:
    """The same seed should reproduce all split assignments."""

    examples = _make_examples(20)

    first = split_examples(
        examples,
        validation_size=4,
        test_size=4,
        seed=42,
    )
    second = split_examples(
        examples,
        validation_size=4,
        test_size=4,
        seed=42,
    )

    assert _example_ids(first.train) == _example_ids(second.train)
    assert _example_ids(first.validation) == _example_ids(second.validation)
    assert _example_ids(first.test) == _example_ids(second.test)


def test_split_examples_contains_no_overlap() -> None:
    """An example must appear in only one split."""

    result = split_examples(
        _make_examples(30),
        validation_size=5,
        test_size=5,
        seed=42,
    )

    train_ids = set(_example_ids(result.train))
    validation_ids = set(_example_ids(result.validation))
    test_ids = set(_example_ids(result.test))

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_split_examples_preserves_every_example() -> None:
    """Splitting must neither lose nor invent records."""

    examples = _make_examples(25)

    result = split_examples(
        examples,
        validation_size=5,
        test_size=5,
        seed=42,
    )

    original_ids = set(_example_ids(examples))
    split_ids = {
        *_example_ids(result.train),
        *_example_ids(result.validation),
        *_example_ids(result.test),
    }

    assert split_ids == original_ids


def test_split_examples_allows_empty_validation_and_test() -> None:
    """All examples may remain in training when holdouts are zero."""

    examples = _make_examples(5)

    result = split_examples(
        examples,
        validation_size=0,
        test_size=0,
    )

    assert result.train_count == 5
    assert result.validation == ()
    assert result.test == ()


def test_split_examples_rejects_excessive_holdout_size() -> None:
    """Validation and test sizes cannot exceed the dataset."""

    with pytest.raises(
        ValueError,
        match="exceeds dataset size",
    ):
        split_examples(
            _make_examples(5),
            validation_size=3,
            test_size=3,
        )


def test_split_examples_rejects_duplicate_ids() -> None:
    """Duplicate IDs would allow train-test leakage."""

    example = _make_examples(1)[0]
    examples = (
        example,
        example,
    )

    with pytest.raises(
        ValueError,
        match="unique example IDs",
    ):
        split_examples(
            examples,
            validation_size=0,
            test_size=1,
        )


def test_dataset_split_rejects_cross_split_duplicate() -> None:
    """DatasetSplit should defend against duplicate IDs."""

    example = _make_examples(1)[0]

    with pytest.raises(
        ValueError,
        match="must not contain duplicate example IDs",
    ):
        DatasetSplit(
            train=(example,),
            validation=(example,),
            test=(),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "validation_size",
        "test_size",
    ],
)
def test_split_examples_rejects_negative_sizes(
    field_name: str,
) -> None:
    """Split sizes must not be negative."""

    arguments = {
        "examples": _make_examples(5),
        "validation_size": 1,
        "test_size": 1,
    }
    arguments[field_name] = -1

    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        split_examples(**arguments)  # type: ignore[arg-type]


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
