"""Tests for normalized dataset statistics."""

import pytest

from pii_scrub.text import CharacterSpan
from research.data.models import DatasetExample
from research.data.statistics import (
    calculate_dataset_statistics,
)


def _make_examples() -> tuple[DatasetExample, ...]:
    """Create examples with multiple sources and entity types."""

    return (
        DatasetExample(
            example_id="openpii-1",
            text="Call John.",
            spans=(CharacterSpan(5, 9, "PERSON"),),
            source="openpii",
            language="en",
        ),
        DatasetExample(
            example_id="openpii-2",
            text="Email jane@example.com.",
            spans=(CharacterSpan(6, 22, "EMAIL"),),
            source="openpii",
            language="en",
        ),
        DatasetExample(
            example_id="gretel-1",
            text="No private information.",
            spans=(),
            source="gretel_finance",
            language="en",
        ),
        DatasetExample(
            example_id="gretel-2",
            text="Jean utilise jean@example.fr.",
            spans=(
                CharacterSpan(0, 4, "PERSON"),
                CharacterSpan(13, 28, "EMAIL"),
            ),
            source="gretel_finance",
            language="fr",
        ),
    )


def test_calculate_dataset_statistics_counts_examples() -> None:
    """Statistics should separate positive and negative examples."""

    result = calculate_dataset_statistics(_make_examples())

    assert result.example_count == 4
    assert result.positive_example_count == 3
    assert result.negative_example_count == 1
    assert result.positive_rate == 0.75


def test_calculate_dataset_statistics_counts_entities() -> None:
    """Statistics should count entities and normalized labels."""

    result = calculate_dataset_statistics(_make_examples())

    assert result.entity_count == 4
    assert result.entities_per_example == 1.0
    assert result.entity_type_counts == {
        "EMAIL": 2,
        "PERSON": 2,
    }


def test_calculate_dataset_statistics_counts_sources() -> None:
    """Every dataset source should have an example count."""

    result = calculate_dataset_statistics(_make_examples())

    assert result.source_counts == {
        "gretel_finance": 2,
        "openpii": 2,
    }


def test_calculate_dataset_statistics_counts_languages() -> None:
    """Every language should have an example count."""

    result = calculate_dataset_statistics(_make_examples())

    assert result.language_counts == {
        "en": 3,
        "fr": 1,
    }


def test_calculate_dataset_statistics_measures_text_length() -> None:
    """Text-length statistics should use character counts."""

    examples = _make_examples()

    result = calculate_dataset_statistics(examples)

    expected_lengths = [len(example.text) for example in examples]

    assert result.total_character_count == sum(expected_lengths)
    assert result.minimum_text_length == min(expected_lengths)
    assert result.maximum_text_length == max(expected_lengths)
    assert result.average_text_length == pytest.approx(
        sum(expected_lengths) / len(expected_lengths)
    )


def test_calculate_dataset_statistics_handles_empty_dataset() -> None:
    """An empty dataset should produce zero-valued statistics."""

    result = calculate_dataset_statistics(())

    assert result.example_count == 0
    assert result.positive_example_count == 0
    assert result.negative_example_count == 0
    assert result.entity_count == 0
    assert result.total_character_count == 0
    assert result.minimum_text_length == 0
    assert result.maximum_text_length == 0
    assert result.average_text_length == 0.0
    assert result.positive_rate == 0.0
    assert result.entities_per_example == 0.0
    assert result.entity_type_counts == {}
    assert result.source_counts == {}
    assert result.language_counts == {}


def test_calculate_dataset_statistics_rejects_list() -> None:
    """Statistics require the immutable tuple representation."""

    with pytest.raises(
        TypeError,
        match="examples must be a tuple",
    ):
        calculate_dataset_statistics(  # type: ignore[arg-type]
            []
        )


def test_calculate_dataset_statistics_rejects_invalid_item() -> None:
    """Every item must be a normalized dataset example."""

    with pytest.raises(
        TypeError,
        match=r"example at index 0 must be a DatasetExample",
    ):
        calculate_dataset_statistics(  # type: ignore[arg-type]
            ("invalid",)
        )


def test_calculate_dataset_statistics_rejects_duplicate_ids() -> None:
    """Duplicate example IDs would distort statistics."""

    example = _make_examples()[0]

    with pytest.raises(
        ValueError,
        match="unique example IDs",
    ):
        calculate_dataset_statistics(
            (
                example,
                example,
            )
        )
