"""Tests for merging predictions from overlapping windows."""

import pytest

from pii_scrub.text import (
    CharacterSpan,
    WindowSpan,
    merge_window_predictions,
    remove_exact_duplicates,
    resolve_cross_type_overlaps,
    resolve_same_type_overlaps,
)


def test_remove_exact_duplicates_keeps_one_prediction() -> None:
    """Identical predictions from two windows should appear once."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
    ]

    result = remove_exact_duplicates(predictions)

    assert result == [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        )
    ]


def test_remove_exact_duplicates_keeps_highest_score() -> None:
    """The most confident duplicate should be retained."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.80,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.95,
        ),
    ]

    result = remove_exact_duplicates(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.95,
        )
    ]


def test_remove_exact_duplicates_keeps_earlier_window_when_scores_equal() -> None:
    """Equal-score duplicates should prefer the earlier window."""

    predictions = [
        WindowSpan(
            window_index=3,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
    ]

    result = remove_exact_duplicates(predictions)

    assert result[0].window_index == 1


def test_remove_exact_duplicates_treats_missing_score_as_lower() -> None:
    """A scored duplicate should beat one without confidence."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.70,
        ),
    ]

    result = remove_exact_duplicates(predictions)

    assert result[0].score == 0.70


def test_remove_exact_duplicates_keeps_different_entity_types() -> None:
    """Same boundaries with different types are not exact duplicates."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "LOCATION"),
            score=0.85,
        ),
    ]

    result = remove_exact_duplicates(predictions)

    assert len(result) == 2

    assert {
        (
            item.span.start,
            item.span.end,
            item.span.entity_type,
        )
        for item in result
    } == {
        (100, 112, "PERSON"),
        (100, 112, "LOCATION"),
    }


def test_remove_exact_duplicates_keeps_different_boundaries() -> None:
    """Different character boundaries are separate predictions."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 113, "PERSON"),
            score=0.85,
        ),
    ]

    result = remove_exact_duplicates(predictions)

    assert result == predictions


def test_remove_exact_duplicates_returns_sorted_output() -> None:
    """Merged predictions should be ordered by document position."""

    predictions = [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(50, 60, "EMAIL"),
            score=0.90,
        ),
        WindowSpan(
            window_index=0,
            span=CharacterSpan(10, 20, "PERSON"),
            score=0.95,
        ),
    ]

    result = remove_exact_duplicates(predictions)

    assert [item.span.start for item in result] == [10, 50]


def test_remove_exact_duplicates_returns_empty_list() -> None:
    """No predictions should produce an empty result."""

    assert remove_exact_duplicates([]) == []


def test_remove_exact_duplicates_rejects_invalid_item() -> None:
    """Every input item must be a WindowSpan."""

    with pytest.raises(
        TypeError,
        match="predictions must contain only WindowSpan objects",
    ):
        remove_exact_duplicates(
            [
                WindowSpan(
                    window_index=0,
                    span=CharacterSpan(0, 4, "PERSON"),
                ),
                "invalid",  # type: ignore[list-item]
            ]
        )


def test_resolve_same_type_overlaps_keeps_highest_score() -> None:
    """Overlapping predictions of one type should prefer confidence."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.80,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 113, "PERSON"),
            score=0.95,
        ),
    ]

    result = resolve_same_type_overlaps(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 113, "PERSON"),
            score=0.95,
        )
    ]


def test_resolve_same_type_overlaps_keeps_longer_span_when_scores_equal() -> None:
    """Equal-confidence overlaps should prefer the longer boundary."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 115, "PERSON"),
            score=0.90,
        ),
    ]

    result = resolve_same_type_overlaps(predictions)

    assert result[0].span == CharacterSpan(100, 115, "PERSON")


def test_resolve_same_type_overlaps_keeps_earlier_window_on_full_tie() -> None:
    """A complete tie should prefer the earlier source window."""

    predictions = [
        WindowSpan(
            window_index=3,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
    ]

    result = resolve_same_type_overlaps(predictions)

    assert result[0].window_index == 1


def test_resolve_same_type_overlaps_keeps_non_overlapping_spans() -> None:
    """Separate spans should both remain."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(10, 20, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(30, 40, "PERSON"),
            score=0.80,
        ),
    ]

    result = resolve_same_type_overlaps(predictions)

    assert result == predictions


def test_resolve_same_type_overlaps_keeps_different_types() -> None:
    """Different entity types are not resolved by this function."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(105, 115, "LOCATION"),
            score=0.95,
        ),
    ]

    result = resolve_same_type_overlaps(predictions)

    assert len(result) == 2
    assert {item.span.entity_type for item in result} == {
        "PERSON",
        "LOCATION",
    }


def test_resolve_same_type_overlaps_treats_touching_spans_as_separate() -> None:
    """Adjacent spans are not overlapping spans."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(10, 20, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(20, 30, "PERSON"),
            score=0.80,
        ),
    ]

    result = resolve_same_type_overlaps(predictions)

    assert len(result) == 2


def test_merge_window_predictions_removes_duplicates_and_overlaps() -> None:
    """The public merge function should apply both merge stages."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.80,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=2,
            span=CharacterSpan(100, 114, "PERSON"),
            score=0.95,
        ),
        WindowSpan(
            window_index=2,
            span=CharacterSpan(200, 220, "EMAIL"),
            score=0.88,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=2,
            span=CharacterSpan(100, 114, "PERSON"),
            score=0.95,
        ),
        WindowSpan(
            window_index=2,
            span=CharacterSpan(200, 220, "EMAIL"),
            score=0.88,
        ),
    ]


def test_merge_window_predictions_resolves_different_type_overlap() -> None:
    """Overlapping different types should keep the better prediction."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(105, 115, "LOCATION"),
            score=0.95,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(105, 115, "LOCATION"),
            score=0.95,
        )
    ]


def test_merge_window_predictions_returns_empty_list() -> None:
    """No window predictions should produce an empty merged result."""

    assert merge_window_predictions([]) == []


def test_merge_window_predictions_returns_sorted_output() -> None:
    """The final merged result should follow document order."""

    predictions = [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(50, 60, "EMAIL"),
            score=0.90,
        ),
        WindowSpan(
            window_index=0,
            span=CharacterSpan(10, 20, "PERSON"),
            score=0.95,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert [item.span.start for item in result] == [10, 50]


def test_resolve_cross_type_overlaps_keeps_highest_score() -> None:
    """Different-type overlaps should prefer higher confidence."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.80,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "LOCATION"),
            score=0.95,
        ),
    ]

    result = resolve_cross_type_overlaps(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "LOCATION"),
            score=0.95,
        )
    ]


def test_resolve_cross_type_overlaps_keeps_longer_span_on_score_tie() -> None:
    """Equal-score conflicts should prefer the longer span."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 115, "LOCATION"),
            score=0.90,
        ),
    ]

    result = resolve_cross_type_overlaps(predictions)

    assert result[0].span == CharacterSpan(
        100,
        115,
        "LOCATION",
    )


def test_resolve_cross_type_overlaps_keeps_earlier_window_on_tie() -> None:
    """Equal score and length should prefer the earlier window."""

    predictions = [
        WindowSpan(
            window_index=3,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "LOCATION"),
            score=0.90,
        ),
    ]

    result = resolve_cross_type_overlaps(predictions)

    assert result[0].window_index == 1
    assert result[0].span.entity_type == "LOCATION"


def test_resolve_cross_type_overlaps_keeps_non_overlapping_types() -> None:
    """Different types should both remain when they do not overlap."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(10, 20, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(30, 40, "LOCATION"),
            score=0.95,
        ),
    ]

    result = resolve_cross_type_overlaps(predictions)

    assert len(result) == 2


def test_resolve_cross_type_overlaps_keeps_touching_spans() -> None:
    """Touching spans are adjacent, not overlapping."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(10, 20, "PERSON"),
            score=0.90,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(20, 30, "LOCATION"),
            score=0.95,
        ),
    ]

    result = resolve_cross_type_overlaps(predictions)

    assert len(result) == 2


def test_merge_window_predictions_resolves_cross_type_conflict() -> None:
    """The public merge function should resolve type disagreements."""

    predictions = [
        WindowSpan(
            window_index=0,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.80,
        ),
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "LOCATION"),
            score=0.95,
        ),
    ]

    result = merge_window_predictions(predictions)

    assert result == [
        WindowSpan(
            window_index=1,
            span=CharacterSpan(100, 112, "LOCATION"),
            score=0.95,
        )
    ]
