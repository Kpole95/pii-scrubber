"""Calibration utilities for scored PII span predictions."""

from collections.abc import Sequence
from dataclasses import dataclass

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.metrics import (
    exact_span_prf,
    expected_calibration_error,
    leak_rate,
    over_redaction_rate,
)


@dataclass(frozen=True, slots=True)
class ThresholdScore:
    """Store aggregate validation metrics for one confidence threshold.

    Example:
        A threshold of ``0.5`` keeps predictions whose score is at least
        ``0.5`` and records the resulting leak and exact-span metrics.
    """

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
    """Mark predictions correct only for exact boundary-and-type matches.

    Example:
        ``PERSON [0, 4)`` is correct only when the gold set contains the
        same start, end, and entity type.
    """

    gold_keys = {
        (
            span.start,
            span.end,
            span.entity_type,
        )
        for span in gold
    }

    return [
        (
            prediction.start,
            prediction.end,
            prediction.entity_type,
        )
        in gold_keys
        for prediction in predictions
    ]


def calibration_error(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    *,
    bins: int = 10,
) -> float:
    """Calculate exact-span expected calibration error across a dataset.

    Example:
        Predictions with confidence close to their observed exact accuracy
        produce a lower ECE.
    """

    _validate_predictions(examples, predictions)

    flat_predictions: list[DetectedSpan] = []
    correctness: list[bool] = []

    for example, output in zip(
        examples,
        predictions,
        strict=True,
    ):
        flat_predictions.extend(output)
        correctness.extend(
            exact_prediction_correctness(
                example.spans,
                output,
            )
        )

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
    """Calculate exact-span ECE independently for each predicted entity.

    Example:
        PERSON and EMAIL receive separate calibration-error values.
    """

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


def threshold_grid(
    *,
    step: float = 0.05,
) -> tuple[float, ...]:
    """Build a deterministic confidence-threshold grid from 0 to 1.

    Example:
        ``step=0.25`` returns ``(0.0, 0.25, 0.5, 0.75, 1.0)``.
    """

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
    """Score one global confidence threshold at a time.

    Example:
        Sweeping ``(0.0, 0.5, 0.9)`` reveals the recall/precision trade-off
        without rerunning model inference.
    """

    _validate_predictions(examples, predictions)

    return tuple(
        _score_threshold(
            examples,
            predictions,
            threshold,
        )
        for threshold in thresholds
    )


def _entity_calibration_error(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    entity: str,
    *,
    bins: int,
) -> float:
    """Calculate ECE for one predicted entity type."""

    entity_predictions: list[DetectedSpan] = []
    correctness: list[bool] = []

    for example, output in zip(
        examples,
        predictions,
        strict=True,
    ):
        selected = [prediction for prediction in output if prediction.entity_type == entity]

        entity_predictions.extend(selected)
        correctness.extend(
            exact_prediction_correctness(
                example.spans,
                selected,
            )
        )

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
            if (prediction.score is None or prediction.score >= threshold)
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

    precision_total = exact_true_positives + exact_false_positives
    recall_total = exact_true_positives + exact_false_negatives

    precision = exact_true_positives / precision_total if precision_total else 0.0

    recall = exact_true_positives / recall_total if recall_total else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return ThresholdScore(
        threshold=threshold,
        leak_rate=(leaked / gold_total if gold_total else 0.0),
        precision=precision,
        recall=recall,
        f1=f1,
        over_redaction_rate=(
            false_positive_characters / non_pii_characters if non_pii_characters else 0.0
        ),
        predictions=prediction_count,
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
            if not isinstance(
                prediction,
                DetectedSpan,
            ):
                raise TypeError(
                    f"prediction at [{output_index}][{prediction_index}] must be a DetectedSpan"
                )
