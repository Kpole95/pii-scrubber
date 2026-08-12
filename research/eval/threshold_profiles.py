"""Select and score per-entity confidence-threshold profiles."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pii_scrub.types import DetectedSpan
from research.data.models import DatasetExample
from research.eval.threshold_metrics import score_filtered_predictions

ThresholdMode = Literal["balanced", "strict"]


@dataclass(frozen=True, slots=True)
class ThresholdProfileScore:
    """Store aggregate metrics for one per-entity threshold profile."""

    leak_rate: float
    precision: float
    recall: float
    f1: float
    over_redaction_rate: float
    predictions: int


def score_threshold_profile(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    thresholds: Mapping[str, float],
) -> ThresholdProfileScore:
    """Score predictions after applying thresholds by entity type."""

    _validate_predictions(examples, predictions)
    _validate_thresholds(thresholds)
    metrics = score_filtered_predictions(
        examples,
        predictions,
        lambda prediction: _passes_threshold(prediction, thresholds),
    )
    return ThresholdProfileScore(
        leak_rate=metrics.leak_rate,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        over_redaction_rate=metrics.over_redaction_rate,
        predictions=metrics.predictions,
    )


def optimize_threshold_profile(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    entities: Sequence[str],
    thresholds: Sequence[float],
    *,
    initial_threshold: float,
    mode: ThresholdMode,
    passes: int = 3,
) -> tuple[dict[str, float], ThresholdProfileScore]:
    """Optimize one threshold per entity with deterministic coordinate search."""

    if mode not in ("balanced", "strict"):
        raise ValueError("mode must be 'balanced' or 'strict'")
    if passes < 1:
        raise ValueError("passes must be at least 1")

    threshold_values = tuple(sorted(set(float(value) for value in thresholds)))
    if not threshold_values:
        raise ValueError("thresholds must not be empty")

    _validate_threshold_value(initial_threshold)
    for threshold in threshold_values:
        _validate_threshold_value(threshold)

    profile = {entity: float(initial_threshold) for entity in sorted(set(entities))}
    for _ in range(passes):
        if not _optimize_pass(
            examples,
            predictions,
            profile,
            threshold_values,
            mode,
        ):
            break

    return profile, score_threshold_profile(examples, predictions, profile)


def _optimize_pass(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    profile: dict[str, float],
    thresholds: Sequence[float],
    mode: ThresholdMode,
) -> bool:
    """Run one coordinate-search pass and report whether it changed."""

    changed = False
    for entity in sorted(profile):
        best_threshold = profile[entity]
        best_score = score_threshold_profile(examples, predictions, profile)

        for threshold in thresholds:
            candidate = dict(profile)
            candidate[entity] = threshold
            score = score_threshold_profile(examples, predictions, candidate)
            if _is_better(
                candidate_score=score,
                candidate_threshold=threshold,
                best_score=best_score,
                best_threshold=best_threshold,
                mode=mode,
            ):
                best_threshold = threshold
                best_score = score

        if best_threshold != profile[entity]:
            profile[entity] = best_threshold
            changed = True

    return changed


def _passes_threshold(
    prediction: DetectedSpan,
    thresholds: Mapping[str, float],
) -> bool:
    """Return whether one prediction survives its entity threshold."""

    if prediction.score is None:
        return True
    return prediction.score >= thresholds.get(prediction.entity_type, 0.0)


def _is_better(
    *,
    candidate_score: ThresholdProfileScore,
    candidate_threshold: float,
    best_score: ThresholdProfileScore,
    best_threshold: float,
    mode: ThresholdMode,
) -> bool:
    """Compare two candidate operating points for the selected mode."""

    if mode == "strict":
        candidate_key = (
            candidate_score.leak_rate,
            -candidate_score.recall,
            candidate_score.over_redaction_rate,
            -candidate_score.precision,
            -candidate_score.f1,
            -candidate_threshold,
        )
        best_key = (
            best_score.leak_rate,
            -best_score.recall,
            best_score.over_redaction_rate,
            -best_score.precision,
            -best_score.f1,
            -best_threshold,
        )
        return candidate_key < best_key

    candidate_key = (
        candidate_score.f1,
        -candidate_score.leak_rate,
        candidate_score.precision,
        candidate_score.recall,
        -candidate_score.over_redaction_rate,
        candidate_threshold,
    )
    best_key = (
        best_score.f1,
        -best_score.leak_rate,
        best_score.precision,
        best_score.recall,
        -best_score.over_redaction_rate,
        best_threshold,
    )
    return candidate_key > best_key


def _validate_predictions(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
) -> None:
    """Validate one prediction collection per example."""

    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal length")


def _validate_thresholds(thresholds: Mapping[str, float]) -> None:
    """Validate entity names and values in a threshold profile."""

    for entity, threshold in thresholds.items():
        if not isinstance(entity, str) or not entity:
            raise ValueError("threshold entity names must be non-empty strings")
        _validate_threshold_value(threshold)


def _validate_threshold_value(threshold: float) -> None:
    """Validate one confidence threshold."""

    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise TypeError("threshold must be numeric")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
