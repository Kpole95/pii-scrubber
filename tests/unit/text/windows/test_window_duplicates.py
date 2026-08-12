"""Tests for removing duplicate predictions from overlapping windows."""

import pytest

from pii_scrub.text import CharacterSpan, WindowSpan, remove_exact_duplicates


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
