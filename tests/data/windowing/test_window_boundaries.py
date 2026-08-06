"""Integration tests for entities predicted near window boundaries."""

from pii_scrub.data.alignment import CharacterSpan
from pii_scrub.data.windowing import (
    WindowSpan,
    merge_window_predictions,
)


def test_same_entity_from_two_windows_becomes_one_span() -> None:
    """An entity repeated in overlapping windows should appear once.

    Example:
        Window 0 and window 1 both detect the same EMAIL.
        The higher-confidence prediction is kept.
    """

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.88,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.94,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.94,
        )
    ]


def test_boundary_disagreement_keeps_best_span() -> None:
    """Slightly different boundaries should keep the better prediction.

    Example:
        One window misses the last character.
        The second window predicts the complete entity with higher confidence.
    """

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(480, 499, "EMAIL"),
            score=0.82,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.95,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.95,
        )
    ]


def test_entity_crossing_window_boundary_is_preserved() -> None:
    """An entity near a window edge must not disappear.

    Both windows see part of the same PERSON entity. The more complete,
    higher-confidence prediction should remain.
    """

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(495, 505, "PERSON"),
            score=0.78,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(495, 512, "PERSON"),
            score=0.93,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(495, 512, "PERSON"),
            score=0.93,
        )
    ]


def test_separate_entities_from_neighboring_windows_are_preserved() -> None:
    """Different non-overlapping entities should both remain."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(450, 462, "PERSON"),
            score=0.91,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(520, 540, "EMAIL"),
            score=0.96,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == predictions


def test_three_windows_repeating_one_entity_produce_one_result() -> None:
    """Three overlapping windows should still produce one final entity."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.80,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.92,
        ),
        WindowSpan(
            window_index=2,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.87,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.92,
        )
    ]


def test_boundary_type_conflict_keeps_best_prediction() -> None:
    """Two windows disagreeing on entity type should keep the best result."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(480, 500, "PERSON"),
            score=0.84,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.97,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.97,
        )
    ]


def test_touching_entities_at_boundary_remain_separate() -> None:
    """Adjacent entities are not duplicates or overlaps.

    Half-open spans (480, 500) and (500, 510) only touch.
    """

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(480, 500, "EMAIL"),
            score=0.94,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(500, 510, "PERSON"),
            score=0.90,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == predictions


def test_boundary_merge_returns_document_order() -> None:
    """Merged entities should be sorted by their document position."""

    predictions = [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(520, 540, "EMAIL"),
            score=0.96,
        ),
        WindowSpan(
            window_index=0,
            span=CharacterSpan(450, 462, "PERSON"),
            score=0.91,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert [prediction.span.start for prediction in result] == [
        450,
        520,
    ]