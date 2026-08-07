"""Span-level metrics for PII redaction systems.

The primary operational metric is leak rate: a gold PII span leaks when any
character remains unredacted. Exact and partial span F1 are reported alongside
it, never as substitutes.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pii_scrub.types import CharacterSpan, DetectedSpan


@dataclass(frozen=True, slots=True)
class PrecisionRecallF1:
    """Store precision, recall, F1, and their integer counts."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def leak_rate(gold: Sequence[CharacterSpan], predicted: Sequence[CharacterSpan]) -> float:
    """Return the fraction of gold spans not fully covered by redaction.

    Entity labels are intentionally ignored: replacing a PERSON span with an
    ADDRESS placeholder is a classification error, but the identifier did not
    leak. Boundary coverage is what matters for this metric.
    """

    _validate_spans(gold, "gold")
    _validate_spans(predicted, "predicted")
    if not gold:
        return 0.0
    leaked = sum(not _fully_covered(target, predicted) for target in gold)
    return leaked / len(gold)


def exact_span_prf(
    gold: Sequence[CharacterSpan], predicted: Sequence[CharacterSpan]
) -> PrecisionRecallF1:
    """Calculate one-to-one exact-boundary, exact-entity span scores."""

    _validate_spans(gold, "gold")
    _validate_spans(predicted, "predicted")
    gold_keys = {(span.start, span.end, span.entity_type) for span in gold}
    predicted_keys = {(span.start, span.end, span.entity_type) for span in predicted}
    true_positives = len(gold_keys & predicted_keys)
    return _prf(
        true_positives,
        len(predicted_keys) - true_positives,
        len(gold_keys) - true_positives,
    )


def partial_span_prf(
    gold: Sequence[CharacterSpan], predicted: Sequence[CharacterSpan]
) -> PrecisionRecallF1:
    """Calculate one-to-one overlap scores for spans of the same entity type.

    Candidate pairs are matched greedily by overlap length, then boundary
    closeness. Each gold and predicted span can contribute to at most one true
    positive.
    """

    _validate_spans(gold, "gold")
    _validate_spans(predicted, "predicted")
    candidates: list[tuple[int, int, int, int]] = []
    for gold_index, gold_span in enumerate(gold):
        for predicted_index, predicted_span in enumerate(predicted):
            if gold_span.entity_type != predicted_span.entity_type:
                continue
            overlap = _overlap_length(gold_span, predicted_span)
            if overlap:
                distance = abs(gold_span.start - predicted_span.start) + abs(
                    gold_span.end - predicted_span.end
                )
                candidates.append((-overlap, distance, gold_index, predicted_index))

    matched_gold: set[int] = set()
    matched_predicted: set[int] = set()
    for _, _, gold_index, predicted_index in sorted(candidates):
        if gold_index in matched_gold or predicted_index in matched_predicted:
            continue
        matched_gold.add(gold_index)
        matched_predicted.add(predicted_index)
    true_positives = len(matched_gold)
    return _prf(
        true_positives,
        len(predicted) - true_positives,
        len(gold) - true_positives,
    )


def per_entity_recall(
    gold: Sequence[CharacterSpan], predicted: Sequence[CharacterSpan]
) -> dict[str, float]:
    """Return exact span recall independently for each gold entity type."""

    entities = sorted({span.entity_type for span in gold})
    return {
        entity: exact_span_prf(
            [span for span in gold if span.entity_type == entity],
            [span for span in predicted if span.entity_type == entity],
        ).recall
        for entity in entities
    }


def over_redaction_rate(
    text_length: int,
    gold: Sequence[CharacterSpan],
    predicted: Sequence[CharacterSpan],
) -> float:
    """Return the fraction of non-PII characters incorrectly redacted."""

    if isinstance(text_length, bool) or not isinstance(text_length, int):
        raise TypeError("text_length must be an integer")
    if text_length < 0:
        raise ValueError("text_length must be non-negative")
    _validate_bounds(gold, text_length, "gold")
    _validate_bounds(predicted, text_length, "predicted")
    gold_chars = _covered_characters(gold)
    non_pii_count = text_length - len(gold_chars)
    if non_pii_count == 0:
        return 0.0
    false_positive_chars = len(_covered_characters(predicted) - gold_chars)
    return false_positive_chars / non_pii_count


def expected_calibration_error(
    predictions: Sequence[DetectedSpan], correctness: Sequence[bool], *, bins: int = 10
) -> float:
    """Calculate equal-width expected calibration error for scored spans."""

    if len(predictions) != len(correctness):
        raise ValueError("predictions and correctness must have equal lengths")
    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an integer")
    if bins <= 0:
        raise ValueError("bins must be greater than zero")
    if not predictions:
        return 0.0
    scores: list[float] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, DetectedSpan):
            raise TypeError(f"prediction at index {index} must be a DetectedSpan")
        if prediction.score is None:
            raise ValueError("ECE requires a score for every prediction")
        scores.append(prediction.score)
    total = len(scores)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, score in enumerate(scores)
            if lower <= score < upper or (bin_index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        confidence = sum(scores[index] for index in members) / len(members)
        accuracy = sum(bool(correctness[index]) for index in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def _fully_covered(target: CharacterSpan, predicted: Sequence[CharacterSpan]) -> bool:
    covered: set[int] = set()
    for span in predicted:
        covered.update(range(max(target.start, span.start), min(target.end, span.end)))
    return len(covered) == target.length


def _overlap_length(left: CharacterSpan, right: CharacterSpan) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def _covered_characters(spans: Sequence[CharacterSpan]) -> set[int]:
    return {position for span in spans for position in range(span.start, span.end)}


def _validate_spans(spans: Sequence[CharacterSpan], name: str) -> None:
    for index, span in enumerate(spans):
        if not isinstance(span, CharacterSpan):
            raise TypeError(f"{name} span at index {index} must be a CharacterSpan")


def _validate_bounds(spans: Sequence[CharacterSpan], text_length: int, name: str) -> None:
    _validate_spans(spans, name)
    for index, span in enumerate(spans):
        if span.end > text_length:
            raise ValueError(f"{name} span at index {index} exceeds text length")


def _prf(true_positives: int, false_positives: int, false_negatives: int) -> PrecisionRecallF1:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PrecisionRecallF1(
        precision,
        recall,
        f1,
        true_positives,
        false_positives,
        false_negatives,
    )
