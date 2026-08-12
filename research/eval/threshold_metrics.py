"""Shared metric helpers for confidence-threshold evaluation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.metrics import exact_span_prf, leak_rate, over_redaction_rate

PredictionFilter = Callable[[DetectedSpan], bool]


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Store aggregate metrics after confidence filtering."""

    leak_rate: float
    precision: float
    recall: float
    f1: float
    over_redaction_rate: float
    predictions: int


def score_filtered_predictions(
    examples: Sequence[DatasetExample],
    predictions: Sequence[Sequence[DetectedSpan]],
    keep_prediction: PredictionFilter,
) -> ThresholdMetrics:
    """Score predictions that pass a caller-provided confidence rule."""

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    leaked = 0
    gold_total = 0
    false_positive_characters = 0
    non_pii_characters = 0
    prediction_count = 0

    for example, output in zip(examples, predictions, strict=True):
        filtered = [prediction for prediction in output if keep_prediction(prediction)]
        prediction_count += len(filtered)
        predicted_spans = [_as_character_span(prediction) for prediction in filtered]

        exact = exact_span_prf(example.spans, predicted_spans)
        true_positives += exact.true_positives
        false_positives += exact.false_positives
        false_negatives += exact.false_negatives

        gold_total += len(example.spans)
        leaked += round(leak_rate(example.spans, predicted_spans) * len(example.spans))

        non_pii = len(example.text) - len(_gold_characters(example))
        non_pii_characters += non_pii
        false_positive_characters += round(
            over_redaction_rate(
                len(example.text),
                example.spans,
                predicted_spans,
            )
            * non_pii
        )

    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _f1(precision, recall)

    return ThresholdMetrics(
        leak_rate=_ratio(leaked, gold_total),
        precision=precision,
        recall=recall,
        f1=f1,
        over_redaction_rate=_ratio(false_positive_characters, non_pii_characters),
        predictions=prediction_count,
    )


def _as_character_span(prediction: DetectedSpan) -> CharacterSpan:
    """Convert one scored prediction into an evaluation span."""

    return CharacterSpan(
        prediction.start,
        prediction.end,
        prediction.entity_type,
    )


def _gold_characters(example: DatasetExample) -> set[int]:
    """Return source positions covered by gold PII spans."""

    return {position for span in example.spans for position in range(span.start, span.end)}


def _ratio(numerator: int, denominator: int) -> float:
    """Return a safe ratio for aggregate metric counts."""

    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """Return F1 for precision and recall values."""

    total = precision + recall
    return 2 * precision * recall / total if total else 0.0
