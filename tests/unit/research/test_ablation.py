"""Tests for encoder evaluation ablations."""

import pytest

from pii_scrub.types import DetectedSpan
from research.eval.ablation import merge_separated_same_type


def test_merges_person_spans_across_whitespace() -> None:
    """Separate name tokens should merge across whitespace."""

    result = merge_separated_same_type(
        "Nadim Ladki",
        [
            DetectedSpan(0, 5, "PERSON", 0.9),
            DetectedSpan(6, 11, "PERSON", 0.8),
        ],
    )

    assert len(result) == 1
    assert result[0].start == 0
    assert result[0].end == 11
    assert result[0].entity_type == "PERSON"
    assert result[0].score == pytest.approx(0.85)


def test_merges_across_punctuation() -> None:
    """Same-type fragments may merge across punctuation separators."""

    result = merge_separated_same_type(
        "Jean-Claude",
        [
            DetectedSpan(0, 4, "PERSON", 0.9),
            DetectedSpan(5, 11, "PERSON", 0.8),
        ],
    )

    assert len(result) == 1
    assert result[0].start == 0
    assert result[0].end == 11
    assert result[0].entity_type == "PERSON"
    assert result[0].score == pytest.approx(0.85)


def test_does_not_merge_across_words() -> None:
    """Lexical text between predictions must block merging."""

    spans = [
        DetectedSpan(0, 4, "PERSON", 0.9),
        DetectedSpan(9, 14, "PERSON", 0.8),
    ]

    assert merge_separated_same_type("John van Smith", spans) == spans


def test_does_not_merge_different_entity_types() -> None:
    """Different entity types must remain separate."""

    spans = [
        DetectedSpan(0, 4, "PERSON", 0.9),
        DetectedSpan(5, 10, "LOCATION", 0.8),
    ]

    assert merge_separated_same_type("John Paris", spans) == spans


def test_empty_predictions_remain_empty() -> None:
    """Empty prediction lists should remain empty."""

    assert merge_separated_same_type("text", []) == []


def test_wrapper_merges_predictor_output() -> None:
    """The ablation wrapper should preserve the predictor contract."""

    def predictor(
        text: str,
        entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        return [
            DetectedSpan(0, 5, "PERSON", 0.9),
            DetectedSpan(6, 11, "PERSON", 0.8),
        ]

    from research.eval.ablation import wrap_merge_ablation

    wrapped = wrap_merge_ablation(predictor)

    result = wrapped("Nadim Ladki", {"PERSON"})

    assert len(result) == 1
    assert result[0].start == 0
    assert result[0].end == 11


def test_can_merge_only_selected_entity_types() -> None:
    """Entity-specific ablations should leave other span types unchanged."""

    result = merge_separated_same_type(
        "John Smith lives New York",
        [
            DetectedSpan(0, 4, "PERSON", 0.9),
            DetectedSpan(5, 10, "PERSON", 0.8),
            DetectedSpan(17, 20, "LOCATION", 0.9),
            DetectedSpan(21, 25, "LOCATION", 0.8),
        ],
        entity_types={"PERSON"},
    )

    assert result[0].start == 0
    assert result[0].end == 10
    assert result[0].entity_type == "PERSON"

    assert result[1:] == [
        DetectedSpan(17, 20, "LOCATION", 0.9),
        DetectedSpan(21, 25, "LOCATION", 0.8),
    ]
