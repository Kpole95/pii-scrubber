"""Build serializable reports for detector baselines."""

from collections.abc import Sequence

from pii_scrub.detectors.base import Detector
from research.data.models import DatasetExample
from research.eval.baselines import DatasetScore, score_dataset
from research.eval.metrics import PrecisionRecallF1


def evaluate_baseline(
    detector: Detector,
    examples: Sequence[DatasetExample],
    *,
    entities: set[str] | None = None,
) -> dict[str, object]:
    """Evaluate one detector and return a serializable report.

    Example:
        A perfect dataset report has exact F1 ``1.0``.
    """
    score = score_dataset(
        detector,
        examples,
        entities=entities,
    )

    return _report(score, len(examples))


def _report(
    score: DatasetScore,
    example_count: int,
) -> dict[str, object]:
    """Convert a dataset score into primitive report values.

    Example:
        ``example_count=2`` is stored as ``{"examples": 2}``.
    """
    return {
        "examples": example_count,
        "leak_rate": score.leak_rate,
        "exact": _prf_report(score.exact),
        "partial": _prf_report(score.partial),
        "over_redaction_rate": score.over_redaction_rate,
        "per_entity_recall": score.per_entity_recall,
    }


def _prf_report(score: PrecisionRecallF1) -> dict[str, object]:
    """Convert one PRF score into primitive values.

    Example:
        Precision ``1.0`` remains a plain float.
    """
    return {
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "true_positives": score.true_positives,
        "false_positives": score.false_positives,
        "false_negatives": score.false_negatives,
    }
