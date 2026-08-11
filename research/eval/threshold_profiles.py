"""Select and score per-entity confidence-threshold profiles."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.metrics import (
    exact_span_prf,
    leak_rate,
    over_redaction_rate,
)

ThresholdMode = Literal["balanced", "strict"]


@dataclass(frozen=True, slots=True)
class ThresholdProfileScore:
    """Store aggregate metrics for one per-entity threshold profile.

    Example:
        PERSON may use ``0.35`` while DATE uses ``0.10``.
        The resulting metrics are calculated over the full dataset.
    """

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
    """Score predictions after applying thresholds by entity type.

    Example:
        ``{"PERSON": 0.4, "EMAIL": 0.7}`` independently filters
        PERSON and EMAIL predictions before aggregate evaluation.
    """

    _validate_predictions(
        examples,
        predictions,
    )
    _validate_thresholds(thresholds)

    exact_true_positives = 0
    exact_false_positives = 0
    exact_false_negatives = 0

    leaked = 0
    gold_total = 0

    false_positive_characters = 0
    non_pii_characters = 0

    prediction_count = 0

    for example, output in zip(
        examples,
        predictions,
        strict=True,
    ):
        filtered = [
            prediction
            for prediction in output
            if _passes_threshold(
                prediction,
                thresholds,
            )
        ]

        prediction_count += len(filtered)

        predicted_spans = [
            CharacterSpan(
                prediction.start,
                prediction.end,
                prediction.entity_type,
            )
            for prediction in filtered
        ]

        exact = exact_span_prf(
            example.spans,
            predicted_spans,
        )

        exact_true_positives += exact.true_positives
        exact_false_positives += exact.false_positives
        exact_false_negatives += exact.false_negatives

        gold_total += len(example.spans)

        leaked += round(
            leak_rate(
                example.spans,
                predicted_spans,
            )
            * len(example.spans)
        )

        gold_characters = {
            position
            for span in example.spans
            for position in range(
                span.start,
                span.end,
            )
        }

        non_pii = len(example.text) - len(gold_characters)

        non_pii_characters += non_pii

        false_positive_characters += round(
            over_redaction_rate(
                len(example.text),
                example.spans,
                predicted_spans,
            )
            * non_pii
        )

    precision_total = (
        exact_true_positives
        + exact_false_positives
    )
    recall_total = (
        exact_true_positives
        + exact_false_negatives
    )

    precision = (
        exact_true_positives / precision_total
        if precision_total
        else 0.0
    )

    recall = (
        exact_true_positives / recall_total
        if recall_total
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return ThresholdProfileScore(
        leak_rate=(
            leaked / gold_total
            if gold_total
            else 0.0
        ),
        precision=precision,
        recall=recall,
        f1=f1,
        over_redaction_rate=(
            false_positive_characters / non_pii_characters
            if non_pii_characters
            else 0.0
        ),
        predictions=prediction_count,
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
    """Optimize one threshold per entity with deterministic coordinate search.

    Example:
        ``mode="strict"`` prioritizes leak rate and recall.
        ``mode="balanced"`` prioritizes exact-span F1.
    """

    if mode not in ("balanced", "strict"):
        raise ValueError(
            "mode must be 'balanced' or 'strict'"
        )

    if passes < 1:
        raise ValueError(
            "passes must be at least 1"
        )

    threshold_values = tuple(
        sorted(set(float(value) for value in thresholds))
    )

    if not threshold_values:
        raise ValueError(
            "thresholds must not be empty"
        )

    _validate_threshold_value(
        initial_threshold,
    )

    for threshold in threshold_values:
        _validate_threshold_value(
            threshold,
        )

    profile = {
        entity: float(initial_threshold)
        for entity in sorted(set(entities))
    }

    for _ in range(passes):
        changed = False

        for entity in sorted(profile):
            best_threshold = profile[entity]
            best_score = score_threshold_profile(
                examples,
                predictions,
                profile,
            )

            for threshold in threshold_values:
                candidate = dict(profile)
                candidate[entity] = threshold

                score = score_threshold_profile(
                    examples,
                    predictions,
                    candidate,
                )

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

        if not changed:
            break

    return (
        profile,
        score_threshold_profile(
            examples,
            predictions,
            profile,
        ),
    )


def _passes_threshold(
    prediction: DetectedSpan,
    thresholds: Mapping[str, float],
) -> bool:
    """Return whether one prediction survives its entity threshold."""

    if prediction.score is None:
        return True

    threshold = thresholds.get(
        prediction.entity_type,
        0.0,
    )

    return prediction.score >= threshold


def _is_better(
    *,
    candidate_score: ThresholdProfileScore,
    candidate_threshold: float,
    best_score: ThresholdProfileScore,
    best_threshold: float,
    mode: ThresholdMode,
) -> bool:
    """Compare two candidate operating points."""

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
        raise ValueError(
            "examples and predictions must have equal length"
        )


def _validate_thresholds(
    thresholds: Mapping[str, float],
) -> None:
    """Validate all values in a threshold profile."""

    for entity, threshold in thresholds.items():
        if not isinstance(entity, str) or not entity:
            raise ValueError(
                "threshold entity names must be non-empty strings"
            )

        _validate_threshold_value(
            threshold,
        )


def _validate_threshold_value(
    threshold: float,
) -> None:
    """Validate one confidence threshold."""

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
    ):
        raise TypeError(
            "threshold must be numeric"
        )

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )