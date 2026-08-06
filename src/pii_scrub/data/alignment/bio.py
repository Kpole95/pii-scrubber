"""Convert word-level BIO labels into character spans."""

from collections.abc import Sequence

from .models import CharacterSpan
from .validation import parse_bio_label
from .words import locate_words


def bio_tags_to_spans(
    text: str,
    words: Sequence[str],
    labels: Sequence[str],
) -> list[CharacterSpan]:
    """Convert strict word-level BIO labels into character spans.

    Example:
        ``Murali/B-PERSON Krishna/I-PERSON`` becomes one PERSON span.
    """

    if len(words) != len(labels):
        raise ValueError(
            "words and labels must contain the same number of items"
        )

    word_offsets = locate_words(text, words)
    parsed_labels = [parse_bio_label(label) for label in labels]

    spans: list[CharacterSpan] = []

    active_start: int | None = None
    active_end: int | None = None
    active_entity_type: str | None = None

    def close_active_span() -> None:
        """Store the current entity and clear its temporary state."""

        nonlocal active_start, active_end, active_entity_type

        if (
            active_start is not None
            and active_end is not None
            and active_entity_type is not None
        ):
            spans.append(
                CharacterSpan(
                    start=active_start,
                    end=active_end,
                    entity_type=active_entity_type,
                )
            )

        active_start = None
        active_end = None
        active_entity_type = None

    for word_index, (word_offset, parsed_label) in enumerate(
        zip(word_offsets, parsed_labels, strict=True)
    ):
        prefix, entity_type = parsed_label

        if prefix == "O":
            close_active_span()
            continue

        if prefix == "B":
            close_active_span()

            active_start = word_offset.start
            active_end = word_offset.end
            active_entity_type = entity_type
            continue

        if active_entity_type is None:
            raise ValueError(
                f"I-label at index {word_index} has no active entity"
            )

        if entity_type != active_entity_type:
            raise ValueError(
                f"I-label at index {word_index} has type "
                f"{entity_type!r}, but active entity has type "
                f"{active_entity_type!r}"
            )

        active_end = word_offset.end

    close_active_span()

    return spans