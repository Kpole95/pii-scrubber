"""Tests for encoder confidence calibration utilities."""

import pytest

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.calibration import (
    calibration_error,
    exact_prediction_correctness,
    per_entity_calibration_error,
    sweep_thresholds,
    threshold_grid,
)


def _example(
    text: str,
    spans: tuple[CharacterSpan, ...],
) -> DatasetExample:
    """Build one compact normalized calibration example."""

    return DatasetExample(
        example_id="example",
        text=text,
        spans=spans,
        source="test",
        language="en",
    )


def test_exact_prediction_correctness_requires_exact_match() -> None:
    """Boundary or entity mistakes should not count as correct."""

    gold = (
        CharacterSpan(
            0,
            4,
            "PERSON",
        ),
    )

    predictions = [
        DetectedSpan(
            0,
            4,
            "PERSON",
            0.9,
        ),
        DetectedSpan(
            0,
            3,
            "PERSON",
            0.8,
        ),
        DetectedSpan(
            0,
            4,
            "ADDRESS",
            0.7,
        ),
    ]

    assert exact_prediction_correctness(
        gold,
        predictions,
    ) == [
        True,
        False,
        False,
    ]


def test_calibration_error_uses_exact_correctness() -> None:
    """Dataset calibration should use exact span correctness."""

    examples = [
        _example(
            "John x",
            (
                CharacterSpan(
                    0,
                    4,
                    "PERSON",
                ),
            ),
        )
    ]

    predictions = [
        [
            DetectedSpan(
                0,
                4,
                "PERSON",
                0.9,
            ),
            DetectedSpan(
                5,
                6,
                "EMAIL",
                0.1,
            ),
        ]
    ]

    assert calibration_error(
        examples,
        predictions,
        bins=2,
    ) == pytest.approx(0.1)


def test_per_entity_calibration_error_separates_entities() -> None:
    """Entity calibration should not mix confidence distributions."""

    examples = [
        _example(
            "John mail",
            (
                CharacterSpan(
                    0,
                    4,
                    "PERSON",
                ),
            ),
        )
    ]

    predictions = [
        [
            DetectedSpan(
                0,
                4,
                "PERSON",
                0.9,
            ),
            DetectedSpan(
                5,
                9,
                "EMAIL",
                0.2,
            ),
        ]
    ]

    result = per_entity_calibration_error(
        examples,
        predictions,
        bins=2,
    )

    assert result["PERSON"] == pytest.approx(0.1)
    assert result["EMAIL"] == pytest.approx(0.2)


def test_threshold_grid_includes_zero_and_one() -> None:
    """Threshold grids should include both endpoints."""

    assert threshold_grid(
        step=0.25,
    ) == (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )


def test_threshold_grid_rejects_invalid_step() -> None:
    """Threshold step must be a numeric value inside the unit interval."""

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        threshold_grid(step=0.0)


def test_threshold_sweep_exposes_precision_recall_tradeoff() -> None:
    """Removing a low-confidence true span should increase leakage."""

    examples = [
        _example(
            "John",
            (
                CharacterSpan(
                    0,
                    4,
                    "PERSON",
                ),
            ),
        ),
        _example(
            "none",
            (),
        ),
    ]

    predictions = [
        [
            DetectedSpan(
                0,
                4,
                "PERSON",
                0.4,
            ),
        ],
        [
            DetectedSpan(
                0,
                4,
                "PERSON",
                0.2,
            ),
        ],
    ]

    scores = sweep_thresholds(
        examples,
        predictions,
        (
            0.0,
            0.3,
            0.5,
        ),
    )

    assert scores[0].leak_rate == 0.0
    assert scores[0].precision == 0.5
    assert scores[0].recall == 1.0

    assert scores[1].leak_rate == 0.0
    assert scores[1].precision == 1.0
    assert scores[1].recall == 1.0

    assert scores[2].leak_rate == 1.0
    assert scores[2].recall == 0.0
    assert scores[2].predictions == 0


def test_threshold_sweep_rejects_mismatched_collections() -> None:
    """Every example must have one prediction collection."""

    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        sweep_thresholds(
            [
                _example(
                    "John",
                    (),
                )
            ],
            [],
            (0.0,),
        )
