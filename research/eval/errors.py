"""Classify span-level prediction errors for evaluation analysis."""

from collections.abc import Sequence
from dataclasses import dataclass

from pii_scrub.types import CharacterSpan, DetectedSpan


@dataclass(frozen=True, slots=True)
class SpanError:
    """Describe one gold span and how predictions relate to it.

    Example:
        A gold ``John Smith`` matched by separate ``John`` and ``Smith``
        predictions is classified as ``split``.
    """

    gold: CharacterSpan
    predictions: tuple[DetectedSpan, ...]
    kind: str


def classify_gold_errors(
    text: str,
    gold_spans: Sequence[CharacterSpan],
    predictions: Sequence[DetectedSpan],
) -> tuple[SpanError, ...]:
    """Classify every gold span that lacks an exact prediction."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    errors: list[SpanError] = []

    for gold in gold_spans:
        if any(_same_span(gold, prediction) for prediction in predictions):
            continue

        overlapping = tuple(prediction for prediction in predictions if _overlaps(gold, prediction))
        errors.append(
            SpanError(
                gold=gold,
                predictions=overlapping,
                kind=_classify(text, gold, overlapping),
            )
        )

    return tuple(errors)


def false_positive_spans(
    gold_spans: Sequence[CharacterSpan],
    predictions: Sequence[DetectedSpan],
) -> tuple[DetectedSpan, ...]:
    """Return predictions that do not overlap any same-type gold span."""

    return tuple(
        prediction
        for prediction in predictions
        if not any(
            gold.entity_type == prediction.entity_type and _overlaps(gold, prediction)
            for gold in gold_spans
        )
    )


def _classify(
    text: str,
    gold: CharacterSpan,
    predictions: Sequence[DetectedSpan],
) -> str:
    """Classify one unmatched gold span from overlapping predictions."""

    if not predictions:
        return "miss"

    same_type = tuple(
        prediction for prediction in predictions if prediction.entity_type == gold.entity_type
    )

    if not same_type:
        return "wrong_type"

    if len(same_type) > 1 and _covers_gold(text, gold, same_type):
        return "split"

    prediction = max(same_type, key=lambda span: _overlap_size(gold, span))

    if prediction.start <= gold.start and prediction.end >= gold.end:
        return "oversized"

    if prediction.start >= gold.start and prediction.end <= gold.end:
        return "undersized"

    return "boundary_shift"


def _same_span(gold: CharacterSpan, prediction: DetectedSpan) -> bool:
    """Return whether type and exact boundaries match."""

    return (
        gold.start == prediction.start
        and gold.end == prediction.end
        and gold.entity_type == prediction.entity_type
    )


def _overlaps(gold: CharacterSpan, prediction: DetectedSpan) -> bool:
    """Return whether two half-open spans share source characters."""

    return gold.start < prediction.end and prediction.start < gold.end


def _covers_gold(
    text: str,
    gold: CharacterSpan,
    predictions: Sequence[DetectedSpan],
) -> bool:
    """Return whether fragments cover gold except non-alphanumeric separators."""

    covered = sorted(
        (
            max(gold.start, prediction.start),
            min(gold.end, prediction.end),
        )
        for prediction in predictions
        if _overlaps(gold, prediction)
    )

    if not covered:
        return False

    _, end = covered[0]
    if covered[0][0] > gold.start:
        return False

    for next_start, next_end in covered[1:]:
        if next_start > end and any(character.isalnum() for character in text[end:next_start]):
            return False
        end = max(end, next_end)

    if end < gold.end:
        return False

    return not any(character.isalnum() for character in text[end : gold.end])


def _overlap_size(
    gold: CharacterSpan,
    prediction: DetectedSpan,
) -> int:
    """Return overlapping character count for two spans."""

    return max(
        0,
        min(gold.end, prediction.end) - max(gold.start, prediction.start),
    )
