"""Tests for window-prediction data models."""

from dataclasses import FrozenInstanceError

import pytest

from pii_scrub.data.alignment import CharacterSpan
from pii_scrub.data.windowing.models import WindowSpan


def test_window_span_stores_prediction() -> None:
    """A valid window prediction should preserve all information."""

    prediction = WindowSpan(
        window_index=2,
        span=CharacterSpan(100, 112, "PERSON"),
        score=0.95,
    )

    assert prediction.window_index == 2
    assert prediction.span == CharacterSpan(100, 112, "PERSON")
    assert prediction.score == 0.95


def test_window_span_allows_missing_score() -> None:
    """Confidence is optional because some detectors do not provide it."""

    prediction = WindowSpan(
        window_index=0,
        span=CharacterSpan(20, 30, "EMAIL"),
    )

    assert prediction.score is None


def test_window_span_converts_integer_score_to_float() -> None:
    """Numeric confidence values should use one consistent float type."""

    prediction = WindowSpan(
        window_index=0,
        span=CharacterSpan(20, 30, "EMAIL"),
        score=1,
    )

    assert prediction.score == 1.0
    assert isinstance(prediction.score, float)


@pytest.mark.parametrize("window_index", [-1, -5])
def test_window_span_rejects_negative_window_index(
    window_index: int,
) -> None:
    """A window index cannot be negative."""

    with pytest.raises(
        ValueError,
        match="window_index must be non-negative",
    ):
        WindowSpan(
            window_index=window_index,
            span=CharacterSpan(0, 4, "PERSON"),
        )


@pytest.mark.parametrize("window_index", [1.5, "1", True])
def test_window_span_rejects_invalid_window_index_type(
    window_index: object,
) -> None:
    """A window index must be a real integer."""

    with pytest.raises(
        TypeError,
        match="window_index must be an integer",
    ):
        WindowSpan(
            window_index=window_index,  # type: ignore[arg-type]
            span=CharacterSpan(0, 4, "PERSON"),
        )


@pytest.mark.parametrize("score", [-0.1, 1.1, 5.0])
def test_window_span_rejects_score_outside_range(
    score: float,
) -> None:
    """Confidence must stay between zero and one."""

    with pytest.raises(
        ValueError,
        match="score must be between 0 and 1",
    ):
        WindowSpan(
            window_index=0,
            span=CharacterSpan(0, 4, "PERSON"),
            score=score,
        )


@pytest.mark.parametrize("score", ["0.9", True])
def test_window_span_rejects_invalid_score_type(
    score: object,
) -> None:
    """Confidence must be numeric or None."""

    with pytest.raises(
        TypeError,
        match="score must be a number or None",
    ):
        WindowSpan(
            window_index=0,
            span=CharacterSpan(0, 4, "PERSON"),
            score=score,  # type: ignore[arg-type]
        )


def test_window_span_is_immutable() -> None:
    """A validated prediction must not change later."""

    prediction = WindowSpan(
        window_index=0,
        span=CharacterSpan(0, 4, "PERSON"),
        score=0.8,
    )

    with pytest.raises(FrozenInstanceError):
        prediction.window_index = 1  # type: ignore[misc]