"""Tests for the Qwen detector adapter."""

import pytest

from pii_scrub.types import DetectedSpan
from research.eval.qwen_detector import QwenDetector


def test_qwen_detector_parses_generated_spans() -> None:
    """Valid sorted Qwen JSON should become detected spans."""

    def generate(_text: str) -> str:
        """Generate deterministic model output for this call."""
        return (
            '{"spans":['
            '{"start":0,"end":4,"entity_type":"PERSON"},'
            '{"start":10,"end":14,"entity_type":"EMAIL"}'
            "]}"
        )

    detector = QwenDetector(generate)

    spans = detector.detect("John text mail")

    assert spans == [
        DetectedSpan(
            start=0,
            end=4,
            entity_type="PERSON",
            score=None,
        ),
        DetectedSpan(
            start=10,
            end=14,
            entity_type="EMAIL",
            score=None,
        ),
    ]
    assert detector.parse_failures == 0


def test_qwen_detector_filters_requested_entities() -> None:
    """Dataset-specific entity filtering should match other detectors."""

    def generate(_text: str) -> str:
        """Generate deterministic model output for this call."""
        return (
            '{"spans":['
            '{"start":0,"end":4,"entity_type":"PERSON"},'
            '{"start":5,"end":12,"entity_type":"EMAIL"}'
            "]}"
        )

    detector = QwenDetector(generate)

    spans = detector.detect(
        "John example",
        entities={"PERSON"},
    )

    assert spans == [
        DetectedSpan(
            start=0,
            end=4,
            entity_type="PERSON",
            score=None,
        )
    ]


def test_qwen_detector_counts_parse_failure() -> None:
    """Invalid JSON should count as a failure and predict no spans."""

    detector = QwenDetector(lambda _text: "not json")

    spans = detector.detect("John")

    assert spans == []
    assert detector.parse_failures == 1


def test_qwen_detector_rejects_unsorted_output() -> None:
    """Unsorted generated spans should remain a strict parse failure."""

    def generate(_text: str) -> str:
        """Generate deterministic model output for this call."""
        return (
            '{"spans":['
            '{"start":5,"end":9,"entity_type":"EMAIL"},'
            '{"start":0,"end":4,"entity_type":"PERSON"}'
            "]}"
        )

    detector = QwenDetector(generate)

    assert detector.detect("John mail") == []
    assert detector.parse_failures == 1


def test_qwen_detector_accumulates_parse_failures() -> None:
    """Parse-failure accounting should accumulate across examples."""

    detector = QwenDetector(lambda _text: "invalid")

    detector.detect("first")
    detector.detect("second")

    assert detector.parse_failures == 2


def test_qwen_detector_enforces_known_entity_types() -> None:
    """Optional taxonomy validation should reject unknown labels."""

    def generate(_text: str) -> str:
        """Generate deterministic model output for this call."""
        return '{"spans":[{"start":0,"end":4,"entity_type":"UNKNOWN"}]}'

    detector = QwenDetector(
        generate,
        entity_types={"PERSON", "EMAIL"},
    )

    assert detector.detect("John") == []
    assert detector.parse_failures == 1


def test_qwen_detector_rejects_non_callable_generator() -> None:
    """Detector construction should reject invalid generators."""

    with pytest.raises(
        TypeError,
        match="generator must be callable",
    ):
        QwenDetector(None)  # type: ignore[arg-type]
