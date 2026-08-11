"""Tests for cached frozen-profile encoder evaluation."""

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from scripts.evaluate_encoder_profiles import _evaluate_cached


def _example() -> DatasetExample:
    return DatasetExample(
        example_id="one",
        text="John public",
        spans=(
            CharacterSpan(
                0,
                4,
                "PERSON",
            ),
        ),
        source="test",
        language="en",
    )


def test_raw_cache_keeps_prediction() -> None:
    examples = (_example(),)

    predictions = (
        (
            DetectedSpan(
                0,
                4,
                "PERSON",
                0.40,
            ),
        ),
    )

    report = _evaluate_cached(
        examples,
        predictions,
        entities=None,
    )

    assert report["exact"]["f1"] == 1.0
    assert report["leak_rate"] == 0.0


def test_thresholded_cache_can_remove_prediction() -> None:
    examples = (_example(),)

    predictions = (
        (
            DetectedSpan(
                0,
                4,
                "PERSON",
                0.40,
            ),
        ),
    )

    report = _evaluate_cached(
        examples,
        predictions,
        thresholds={
            "PERSON": 0.50,
        },
        entities=None,
    )

    assert report["exact"]["recall"] == 0.0
    assert report["leak_rate"] == 1.0


def test_entity_filter_is_preserved() -> None:
    examples = (_example(),)

    predictions = (
        (
            DetectedSpan(
                0,
                4,
                "PERSON",
                0.90,
            ),
            DetectedSpan(
                5,
                11,
                "EMAIL",
                0.90,
            ),
        ),
    )

    report = _evaluate_cached(
        examples,
        predictions,
        entities={"PERSON"},
    )

    assert report["exact"]["precision"] == 1.0
    assert report["exact"]["recall"] == 1.0