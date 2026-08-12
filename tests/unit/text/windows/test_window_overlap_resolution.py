"""Tests for resolving overlapping predictions from neighboring windows."""

from pii_scrub.text import (
    CharacterSpan,
    WindowSpan,
    resolve_cross_type_overlaps,
    resolve_same_type_overlaps,
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
