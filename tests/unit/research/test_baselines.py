"""Tests for detector baseline evaluation helpers."""

from pii_scrub.detectors.base import Detector
from pii_scrub.detectors.presidio import PresidioDetector
from pii_scrub.detectors.regex import RegexDetector
from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.baselines import (
    predict_spans,
    score_dataset,
    score_example,
)


class FakeDetector(Detector):
    """Return one predictable EMAIL span for baseline tests."""

    def detect(
        self,
        text: str,
        *,
        entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        """Return the known email span.

        Example:
            ``"Mail a@b.com"`` returns the EMAIL at ``[5, 12)``.
        """
        return [DetectedSpan(5, 12, "EMAIL", 1.0)]


def test_predict_spans_normalizes_detector_output() -> None:
    """Detector predictions should become CharacterSpan objects."""
    example = DatasetExample(
        example_id="test-1",
        text="Mail a@b.com",
        spans=(CharacterSpan(5, 12, "EMAIL"),),
        source="test",
        language="en",
    )

    assert predict_spans(FakeDetector(), example) == (CharacterSpan(5, 12, "EMAIL"),)


def test_predict_spans_with_regex_detector() -> None:
    """The real regex detector should work through the baseline bridge."""
    example = DatasetExample(
        example_id="test-2",
        text="Email me at user@example.com",
        spans=(CharacterSpan(12, 28, "EMAIL"),),
        source="test",
        language="en",
    )

    assert predict_spans(RegexDetector(), example) == (CharacterSpan(12, 28, "EMAIL"),)


def test_predict_spans_with_presidio_detector() -> None:
    """Presidio should work through the same baseline bridge."""
    example = DatasetExample(
        example_id="test-3",
        text="Email me at user@example.com",
        spans=(CharacterSpan(12, 28, "EMAIL"),),
        source="test",
        language="en",
    )

    result = predict_spans(PresidioDetector(), example)

    assert CharacterSpan(12, 28, "EMAIL") in result


def test_score_example_uses_shared_metrics() -> None:
    """A perfect detector prediction should score perfectly."""
    example = DatasetExample(
        example_id="test-score",
        text="Mail a@b.com",
        spans=(CharacterSpan(5, 12, "EMAIL"),),
        source="test",
        language="en",
    )

    score = score_example(
        FakeDetector(),
        example,
    )

    assert score.leak_rate == 0.0
    assert score.exact.f1 == 1.0
    assert score.partial.f1 == 1.0
    assert score.over_redaction_rate == 0.0


def test_score_dataset_aggregates_counts() -> None:
    """Dataset scoring should aggregate TP, FP, and FN."""
    examples = (
        DatasetExample(
            example_id="one",
            text="Mail a@b.com",
            spans=(CharacterSpan(5, 12, "EMAIL"),),
            source="test",
            language="en",
        ),
        DatasetExample(
            example_id="two",
            text="Mail a@b.com",
            spans=(CharacterSpan(5, 12, "EMAIL"),),
            source="test",
            language="en",
        ),
    )

    score = score_dataset(
        FakeDetector(),
        examples,
    )

    assert score.exact.true_positives == 2
    assert score.exact.false_positives == 0
    assert score.exact.false_negatives == 0
    assert score.exact.f1 == 1.0
    assert score.leak_rate == 0.0


def test_score_dataset_keeps_entities_example_aware() -> None:
    """Identical offsets in different examples must not cross-match."""
    examples = (
        DatasetExample(
            example_id="one",
            text="Mail a@b.com",
            spans=(CharacterSpan(5, 12, "EMAIL"),),
            source="test",
            language="en",
        ),
        DatasetExample(
            example_id="two",
            text="Nothing here",
            spans=(CharacterSpan(5, 12, "PERSON"),),
            source="test",
            language="en",
        ),
    )

    score = score_dataset(
        FakeDetector(),
        examples,
    )

    assert score.per_entity_recall["EMAIL"] == 1.0
    assert score.per_entity_recall["PERSON"] == 0.0
