"""Tests for reusable baseline reports."""

from pii_scrub.detectors.base import Detector
from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.run_baseline import evaluate_baseline


class PerfectDetector(Detector):
    """Return the expected EMAIL span.

    Example:
        ``"Mail a@b.com"`` returns ``[5, 12)``.
    """

    def detect(
        self,
        text: str,
        *,
        entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        """Return one fixed test prediction.

        Example:
            The result is one EMAIL span.
        """
        return [
            DetectedSpan(
                5,
                12,
                "EMAIL",
                1.0,
            )
        ]


def test_evaluate_baseline_returns_report() -> None:
    """Baseline evaluation should return primitive report data."""
    examples = (
        DatasetExample(
            example_id="one",
            text="Mail a@b.com",
            spans=(
                CharacterSpan(
                    5,
                    12,
                    "EMAIL",
                ),
            ),
            source="test",
            language="en",
        ),
    )

    report = evaluate_baseline(
        PerfectDetector(),
        examples,
    )

    assert report["examples"] == 1
    assert report["leak_rate"] == 0.0
    assert report["over_redaction_rate"] == 0.0
    assert report["per_entity_recall"] == {
        "EMAIL": 1.0,
    }

    exact = report["exact"]

    assert isinstance(exact, dict)
    assert exact["f1"] == 1.0
    assert exact["true_positives"] == 1
