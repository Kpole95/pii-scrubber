"""Calibration utilities for scored PII span predictions."""

from collections.abc import Sequence
from dataclasses import dataclass

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.metrics import expected_calibration_error
from research.eval.threshold_metrics import score_filtered_predictions


@dataclass(frozen=True, slots=True)
class ThresholdScore:
    """Store aggregate validation metrics for one confidence threshold."""

    threshold: float
    leak_rate: float
    precision: float
    recall: float
    f1: float
    over_redaction_rate: float
    predictions: int


def exact_prediction_correctness(
    gold: Sequence[CharacterSpan],
    predictions: Sequence[DetectedSpan],
) -> list[bool]:
    """Mark predictions correct only for exact boundary-and-type matches."""

    gold_keys = {(span.start, span.end, span.entity_type) for span in gold}
    return [
        (prediction.start, prediction.end, prediction.entity_type) in gold_keys
        for prediction in predictions
    ]


def calibration_error(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    *,
    bins: int = 10,
) -> float:
    """Calculate exact-span expected calibration error for a dataset."""

    _validate_predictions(examples, predictions)
    flat_predictions: list[DetectedSpan] = []
    correctness: list[bool] = []

    for example, output in zip(examples, predictions, strict=True):
        flat_predictions.extend(output)
        correctness.extend(exact_prediction_correctness(example.spans, output))

    return expected_calibration_error(
        flat_predictions,
        correctness,
        bins=bins,
    )


def per_entity_calibration_error(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    *,
    bins: int = 10,
) -> dict[str, float]:
    """Calculate exact-span calibration error for each predicted entity."""

    _validate_predictions(examples, predictions)
    entities = sorted({prediction.entity_type for output in predictions for prediction in output})
    return {
        entity: _entity_calibration_error(
            examples,
            predictions,
            entity,
            bins=bins,
        )
        for entity in entities
    }


def threshold_grid(*, step: float = 0.05) -> tuple[float, ...]:
    """Build a deterministic confidence-threshold grid from zero to one."""

    if isinstance(step, bool) or not isinstance(step, int | float):
        raise TypeError("step must be numeric")

    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be greater than 0 and at most 1")

    values: list[float] = []
    index = 0
    while index * step < 1.0:
        values.append(round(index * step, 10))
        index += 1
    values.append(1.0)
    return tuple(values)


def sweep_thresholds(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    thresholds: Sequence[float],
) -> tuple[ThresholdScore, ...]:
    """Score each global confidence threshold without rerunning inference."""

    _validate_predictions(examples, predictions)
    return tuple(_score_threshold(examples, predictions, threshold) for threshold in thresholds)


def _entity_calibration_error(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    entity: str,
    *,
    bins: int,
) -> float:
    """Calculate calibration error for one predicted entity type."""

    entity_predictions: list[DetectedSpan] = []
    correctness: list[bool] = []

    for example, output in zip(examples, predictions, strict=True):
        selected = [prediction for prediction in output if prediction.entity_type == entity]
        entity_predictions.extend(selected)
        correctness.extend(exact_prediction_correctness(example.spans, selected))

    return expected_calibration_error(
        entity_predictions,
        correctness,
        bins=bins,
    )


def _score_threshold(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    threshold: float,
) -> ThresholdScore:
    """Calculate aggregate metrics after one global threshold."""

    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise TypeError("threshold must be numeric")

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    metrics = score_filtered_predictions(
        examples,
        predictions,
        lambda prediction: prediction.score is None or prediction.score >= threshold,
    )
    return ThresholdScore(
        threshold=threshold,
        leak_rate=metrics.leak_rate,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        over_redaction_rate=metrics.over_redaction_rate,
        predictions=metrics.predictions,
    )


def _validate_predictions(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
) -> None:
    """Validate one prediction collection per example."""

    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal lengths")

    for output_index, output in enumerate(predictions):
        for prediction_index, prediction in enumerate(output):
            if not isinstance(prediction, DetectedSpan):
                raise TypeError(
                    f"prediction at [{output_index}][{prediction_index}] must be a DetectedSpan"
                )
