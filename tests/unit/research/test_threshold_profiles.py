"""Tests for per-entity threshold profile scoring."""

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.threshold_profiles import (
    optimize_threshold_profile,
    score_threshold_profile,
)


def _example() -> DatasetExample:
    return DatasetExample(
        example_id="example-1",
        source="test",
        language="en",
        text="John Smith public",
        spans=(
            CharacterSpan(
                start=0,
                end=10,
                entity_type="PERSON",
            ),
        ),
    )


def test_profile_threshold_filters_prediction() -> None:
    examples = (_example(),)

    predictions = (
        (
            DetectedSpan(
                start=0,
                end=10,
                entity_type="PERSON",
                score=0.40,
            ),
        ),
    )

    kept = score_threshold_profile(
        examples,
        predictions,
        {"PERSON": 0.35},
    )

    removed = score_threshold_profile(
        examples,
        predictions,
        {"PERSON": 0.45},
    )

    assert kept.recall == 1.0
    assert kept.leak_rate == 0.0

    assert removed.recall == 0.0
    assert removed.leak_rate == 1.0


def test_none_score_is_always_kept() -> None:
    examples = (_example(),)

    predictions = (
        (
            DetectedSpan(
                start=0,
                end=10,
                entity_type="PERSON",
                score=None,
            ),
        ),
    )

    score = score_threshold_profile(
        examples,
        predictions,
        {"PERSON": 1.0},
    )

    assert score.recall == 1.0
    assert score.predictions == 1


def test_strict_optimizer_prefers_lower_leak() -> None:
    examples = (_example(),)

    predictions = (
        (
            DetectedSpan(
                start=0,
                end=10,
                entity_type="PERSON",
                score=0.40,
            ),
        ),
    )

    profile, score = optimize_threshold_profile(
        examples,
        predictions,
        ("PERSON",),
        (0.0, 0.5, 1.0),
        initial_threshold=0.5,
        mode="strict",
        passes=2,
    )

    assert profile["PERSON"] == 0.0
    assert score.leak_rate == 0.0
    assert score.recall == 1.0


def test_balanced_optimizer_removes_false_positive() -> None:
    examples = (_example(),)

    predictions = (
        (
            DetectedSpan(
                start=0,
                end=10,
                entity_type="PERSON",
                score=0.90,
            ),
            DetectedSpan(
                start=11,
                end=17,
                entity_type="PERSON",
                score=0.20,
            ),
        ),
    )

    profile, score = optimize_threshold_profile(
        examples,
        predictions,
        ("PERSON",),
        (0.0, 0.5, 1.0),
        initial_threshold=0.0,
        mode="balanced",
        passes=2,
    )

    assert profile["PERSON"] == 0.5
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_mismatched_prediction_collections_fail() -> None:
    examples = (_example(),)

    try:
        score_threshold_profile(
            examples,
            (),
            {},
        )
    except ValueError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError"
        )