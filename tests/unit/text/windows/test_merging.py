"""Tests for the public window-prediction merge pipeline."""

from pii_scrub.text import CharacterSpan, WindowSpan, merge_window_predictions


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
