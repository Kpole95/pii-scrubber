"""Convert BIO annotations and token predictions into character spans."""

from collections.abc import Sequence
from typing import Literal

from pii_scrub.types import AlignedExample, CharacterSpan, DetectedSpan, Offset

BioPrefix = Literal["B", "I", "O"]


def parse_bio_label(label: str) -> tuple[BioPrefix, str | None]:
    """Parse one strict BIO label.

    Examples:
        ``"O"`` becomes ``("O", None)`` and ``"B-PERSON"`` becomes
        ``("B", "PERSON")``.
    """

    if not isinstance(label, str):
        raise TypeError("BIO label must be a string")
    if label == "O":
        return "O", None
    if label != label.strip() or "-" not in label:
        raise ValueError(f"malformed BIO label: {label!r}")
    prefix, entity_type = label.split("-", 1)
    if prefix not in {"B", "I"} or not entity_type or entity_type != entity_type.strip():
        raise ValueError(f"malformed BIO label: {label!r}")
    if prefix == "B":
        return "B", entity_type
    return "I", entity_type


def bio_tags_to_spans(
    text: str, words: Sequence[str], labels: Sequence[str]
) -> list[CharacterSpan]:
    """Convert strict word-level BIO labels into source character spans."""

    from pii_scrub.text.alignment import locate_words

    if len(words) != len(labels):
        raise ValueError("words and labels must contain the same number of items")
    offsets = locate_words(text, words)
    spans: list[CharacterSpan] = []
    active: tuple[int, int, str] | None = None

    def close() -> None:
        nonlocal active
        if active is not None:
            spans.append(CharacterSpan(active[0], active[1], active[2]))
        active = None

    for index, (offset, label) in enumerate(zip(offsets, labels, strict=True)):
        prefix, entity_type = parse_bio_label(label)
        if prefix == "O":
            close()
        elif prefix == "B":
            close()
            active = (offset.start, offset.end, entity_type or "")
        elif active is None:
            raise ValueError(f"I-label at index {index} has no active entity")
        elif entity_type != active[2]:
            raise ValueError(
                f"I-label at index {index} has type {entity_type!r}, "
                f"but active entity has type {active[2]!r}"
            )
        else:
            active = (active[0], offset.end, active[2])
    close()
    return spans


def aligned_labels_to_spans(example: AlignedExample) -> list[CharacterSpan]:
    """Reconstruct source spans from aligned token BIO labels.

    Ignored continuation pieces extend the active word, which preserves the
    full entity under the ``first_subword`` training strategy.
    """

    if not isinstance(example, AlignedExample):
        raise TypeError("example must be an AlignedExample")
    spans: list[CharacterSpan] = []
    active: tuple[int, int, str, int | None] | None = None

    def close() -> None:
        nonlocal active
        if active is not None:
            spans.append(CharacterSpan(active[0], active[1], active[2]))
        active = None

    for index, (offset, word_id, label) in enumerate(
        zip(example.offset_mapping, example.word_ids, example.token_labels, strict=True)
    ):
        start, end = offset
        if label is None:
            if active is not None and word_id is not None and word_id == active[3]:
                active = (active[0], end, active[2], active[3])
            continue
        prefix, entity_type = parse_bio_label(label)
        if prefix == "O":
            close()
        elif prefix == "B":
            close()
            active = (start, end, entity_type or "", word_id)
        elif active is None:
            raise ValueError(f"I-label at token index {index} has no active entity")
        elif entity_type != active[2]:
            raise ValueError(
                f"I-label at token index {index} has type {entity_type!r}, "
                f"but active entity has type {active[2]!r}"
            )
        else:
            active = (active[0], end, active[2], word_id)
    close()
    return spans


def token_predictions_to_spans(
    offsets: Sequence[Offset],
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
) -> list[DetectedSpan]:
    """Decode model BIO predictions into scored character spans.

    Invalid ``I`` transitions start a new entity instead of crashing inference.
    Zero-length offsets from special or padding tokens are ignored. When token
    probabilities are supplied, each span receives their arithmetic mean.

    Example:
        ``B-PERSON`` + ``I-PERSON`` over ``[5, 8)`` and ``[8, 11)`` becomes
        ``DetectedSpan(5, 11, "PERSON", score=...)``.
    """

    if len(offsets) != len(labels):
        raise ValueError("offsets and labels must contain the same number of items")
    if scores is not None and len(scores) != len(labels):
        raise ValueError("scores and labels must contain the same number of items")

    spans: list[DetectedSpan] = []
    active: tuple[int, int, str, list[float]] | None = None

    def close() -> None:
        nonlocal active
        if active is not None:
            start, end, entity_type, probabilities = active
            score = sum(probabilities) / len(probabilities) if probabilities else None
            spans.append(DetectedSpan(start, end, entity_type, score))
        active = None

    for index, (offset, label) in enumerate(zip(offsets, labels, strict=True)):
        if (
            not isinstance(offset, tuple)
            or len(offset) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offset)
        ):
            raise TypeError(f"offset at index {index} must contain two integers")

        start, end = offset
        if start < 0 or end < start:
            raise ValueError(f"offset at index {index} is invalid")

        probability = None if scores is None else scores[index]
        if probability is not None:
            if isinstance(probability, bool) or not isinstance(probability, int | float):
                raise TypeError(f"score at index {index} must be a number")
            probability = float(probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"score at index {index} must be between 0 and 1")

        prefix, entity_type = parse_bio_label(label)

        if start == end:
            continue

        if prefix == "O":
            close()
            continue

        token_scores = [] if probability is None else [probability]

        if prefix == "B" or active is None or entity_type != active[2]:
            close()
            active = (start, end, entity_type or "", token_scores)
            continue

        active = (
            min(active[0], start),
            max(active[1], end),
            active[2],
            [*active[3], *token_scores],
        )

    close()
    return spans
