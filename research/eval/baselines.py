"""Helpers for evaluating runtime detectors as research baselines."""

from collections.abc import Sequence
from dataclasses import dataclass

from pii_scrub.detectors.base import Detector
from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.eval.metrics import (
    PrecisionRecallF1,
    exact_span_prf,
    leak_rate,
    over_redaction_rate,
    partial_span_prf,
)


@dataclass(frozen=True, slots=True)
class BaselineScore:
    """Store metrics for one evaluated example.

    Example:
        A perfect prediction has leak rate ``0.0`` and exact F1 ``1.0``.
    """

    leak_rate: float
    exact: PrecisionRecallF1
    partial: PrecisionRecallF1
    over_redaction_rate: float


@dataclass(frozen=True, slots=True)
class DatasetScore:
    """Store aggregate metrics for a dataset.

    Example:
        Perfect predictions produce exact F1 and entity recall of ``1.0``.
    """

    leak_rate: float
    exact: PrecisionRecallF1
    partial: PrecisionRecallF1
    over_redaction_rate: float
    per_entity_recall: dict[str, float]


def score_dataset(
    detector: Detector,
    examples: Sequence[DatasetExample],
    *,
    entities: set[str] | None = None,
) -> DatasetScore:
    """Score a detector across multiple examples.

    Example:
        Perfect predictions across all rows produce F1 ``1.0``.
    """
    predicted = [
        predict_spans(
            detector,
            example,
            entities=entities,
        )
        for example in examples
    ]

    gold = [example.spans for example in examples]

    exact = [
        exact_span_prf(target, output)
        for target, output in zip(
            gold,
            predicted,
            strict=True,
        )
    ]

    partial = [
        partial_span_prf(target, output)
        for target, output in zip(
            gold,
            predicted,
            strict=True,
        )
    ]

    return DatasetScore(
        leak_rate=_dataset_leak_rate(
            gold,
            predicted,
        ),
        exact=_merge_prf(exact),
        partial=_merge_prf(partial),
        over_redaction_rate=_dataset_over_redaction(
            examples,
            predicted,
        ),
        per_entity_recall=_dataset_entity_recall(
            examples,
            predicted,
        ),
    )


def _dataset_entity_recall(
    examples: Sequence[DatasetExample],
    predicted: list[tuple[CharacterSpan, ...]],
) -> dict[str, float]:
    """Calculate exact recall per entity across examples.

    Example:
        One correct PERSON span out of two gives recall ``0.5``.
    """
    entities = sorted({span.entity_type for example in examples for span in example.spans})

    return {
        entity: _entity_recall(
            examples,
            predicted,
            entity,
        )
        for entity in entities
    }


def _entity_recall(
    examples: Sequence[DatasetExample],
    predicted: list[tuple[CharacterSpan, ...]],
    entity: str,
) -> float:
    """Aggregate exact recall for one entity.

    Example:
        Two true positives from four gold spans give ``0.5``.
    """
    true_positives = 0
    false_negatives = 0

    for example, output in zip(
        examples,
        predicted,
        strict=True,
    ):
        gold = [span for span in example.spans if span.entity_type == entity]
        found = [span for span in output if span.entity_type == entity]
        score = exact_span_prf(
            gold,
            found,
        )
        true_positives += score.true_positives
        false_negatives += score.false_negatives

    total = true_positives + false_negatives
    return true_positives / total if total else 0.0


def _merge_prf(
    scores: list[PrecisionRecallF1],
) -> PrecisionRecallF1:
    """Combine metric counts before calculating PRF.

    Example:
        Counts from many examples become one dataset score.
    """
    true_positives = sum(score.true_positives for score in scores)
    false_positives = sum(score.false_positives for score in scores)
    false_negatives = sum(score.false_negatives for score in scores)

    precision_total = true_positives + false_positives
    recall_total = true_positives + false_negatives

    precision = true_positives / precision_total if precision_total else 0.0
    recall = true_positives / recall_total if recall_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return PrecisionRecallF1(
        precision,
        recall,
        f1,
        true_positives,
        false_positives,
        false_negatives,
    )


def _dataset_leak_rate(
    gold: list[tuple[CharacterSpan, ...]],
    predicted: list[tuple[CharacterSpan, ...]],
) -> float:
    """Calculate leak rate across all gold spans.

    Example:
        One leaked span out of two gives ``0.5``.
    """
    leaked = 0
    total = 0

    for target, output in zip(
        gold,
        predicted,
        strict=True,
    ):
        total += len(target)
        leaked += round(leak_rate(target, output) * len(target))

    return leaked / total if total else 0.0


def predict_spans(
    detector: Detector,
    example: DatasetExample,
    *,
    entities: set[str] | None = None,
) -> tuple[CharacterSpan, ...]:
    """Run a detector and return normalized spans.

    Example:
        A detected EMAIL at ``[5, 12)`` becomes a ``CharacterSpan``.
    """
    predictions = detector.detect(
        example.text,
        entities=entities,
    )

    return tuple(
        CharacterSpan(
            span.start,
            span.end,
            span.entity_type,
        )
        for span in predictions
    )


def _dataset_over_redaction(
    examples: Sequence[DatasetExample],
    predicted: list[tuple[CharacterSpan, ...]],
) -> float:
    """Weight over-redaction by non-PII characters.

    Example:
        Larger examples contribute proportionally more characters.
    """
    false_redacted = 0.0
    non_pii = 0

    for example, output in zip(
        examples,
        predicted,
        strict=True,
    ):
        gold_chars = sum(span.length for span in example.spans)
        available = len(example.text) - gold_chars

        false_redacted += (
            over_redaction_rate(
                len(example.text),
                example.spans,
                output,
            )
            * available
        )
        non_pii += available

    return false_redacted / non_pii if non_pii else 0.0


def score_example(
    detector: Detector,
    example: DatasetExample,
    *,
    entities: set[str] | None = None,
) -> BaselineScore:
    """Score one example with the shared evaluation metrics.

    Example:
        Exact gold and prediction spans produce exact F1 ``1.0``.
    """
    predicted = predict_spans(
        detector,
        example,
        entities=entities,
    )
    gold = example.spans

    return BaselineScore(
        leak_rate=leak_rate(gold, predicted),
        exact=exact_span_prf(gold, predicted),
        partial=partial_span_prf(gold, predicted),
        over_redaction_rate=over_redaction_rate(
            len(example.text),
            gold,
            predicted,
        ),
    )
