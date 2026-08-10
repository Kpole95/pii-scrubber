"""Prediction post-processing ablations for encoder evaluation."""

from collections.abc import Sequence

from pii_scrub.types import DetectedSpan
from research.eval.encoder import EncoderPredictor


def merge_separated_same_type(
    text: str,
    spans: Sequence[DetectedSpan],
    *,
    entity_types: set[str] | None = None,
) -> list[DetectedSpan]:
    """Merge selected same-type spans separated only by non-alphanumeric text.

    Example:
        ``PERSON John`` + space + ``PERSON Smith`` becomes one PERSON span.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[DetectedSpan] = [ordered[0]]

    for span in ordered[1:]:
        previous = merged[-1]
        gap = text[previous.end : span.start]
        selected = entity_types is None or previous.entity_type in entity_types

        if (
            selected
            and previous.entity_type == span.entity_type
            and span.start >= previous.end
            and not any(character.isalnum() for character in gap)
        ):
            scores = [score for score in (previous.score, span.score) if score is not None]
            merged[-1] = DetectedSpan(
                previous.start,
                span.end,
                previous.entity_type,
                sum(scores) / len(scores) if scores else None,
            )
            continue

        merged.append(span)

    return merged


def wrap_merge_ablation(
    predictor: EncoderPredictor,
    *,
    entity_types: set[str] | None = None,
) -> EncoderPredictor:
    """Wrap an encoder predictor with selected same-type fragment merging."""

    def predict(
        text: str,
        entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        return merge_separated_same_type(
            text,
            predictor(text, entities),
            entity_types=entity_types,
        )

    return predict
