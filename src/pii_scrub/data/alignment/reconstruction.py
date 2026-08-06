"""Reconstruct character entities from aligned token BIO labels."""

from .models import AlignedExample, CharacterSpan
from .validation import parse_bio_label


def aligned_labels_to_spans(
    example: AlignedExample,
) -> list[CharacterSpan]:
    """Reconstruct character spans from aligned token labels.

    Ignored continuation subwords can still extend the active word boundary.

    Example:
        ``Mur/B-PERSON`` and ``##ali/None`` reconstruct ``"Murali"``
        when both tokens have the same word ID.
    """

    if not isinstance(example, AlignedExample):
        raise TypeError("example must be an AlignedExample")

    spans: list[CharacterSpan] = []

    active_start: int | None = None
    active_end: int | None = None
    active_entity_type: str | None = None
    active_word_id: int | None = None

    def close_active_span() -> None:
        """Store the active entity and clear its state."""

        nonlocal active_start
        nonlocal active_end
        nonlocal active_entity_type
        nonlocal active_word_id

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
        active_word_id = None

    for token_index, (offset, word_id, label) in enumerate(
        zip(
            example.offset_mapping,
            example.word_ids,
            example.token_labels,
            strict=True,
        )
    ):
        start, end = offset

        if label is None:
            if (
                active_entity_type is not None
                and word_id is not None
                and word_id == active_word_id
            ):
                active_end = end

            continue

        prefix, entity_type = parse_bio_label(label)

        if prefix == "O":
            close_active_span()
            continue

        if prefix == "B":
            close_active_span()

            active_start = start
            active_end = end
            active_entity_type = entity_type
            active_word_id = word_id
            continue

        if active_entity_type is None:
            raise ValueError(
                f"I-label at token index {token_index} "
                "has no active entity"
            )

        if entity_type != active_entity_type:
            raise ValueError(
                f"I-label at token index {token_index} has type "
                f"{entity_type!r}, but active entity has type "
                f"{active_entity_type!r}"
            )

        active_end = end
        active_word_id = word_id

    close_active_span()

    return spans