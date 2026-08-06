"""Validation and label-conversion helpers."""

from collections.abc import Mapping, Sequence

from .constants import IGNORE_INDEX
from .types import BioPrefix, TokenLabel


def parse_bio_label(
    label: str,
) -> tuple[BioPrefix, str | None]:
    """Parse one strict BIO label.

    Examples:
        ``"O"`` returns ``("O", None)``.
        ``"B-PERSON"`` returns ``("B", "PERSON")``.
    """

    if not isinstance(label, str):
        raise TypeError("BIO label must be a string")

    if label == "O":
        return "O", None

    if label != label.strip():
        raise ValueError(f"malformed BIO label: {label!r}")

    if "-" not in label:
        raise ValueError(f"malformed BIO label: {label!r}")

    prefix, entity_type = label.split("-", maxsplit=1)

    if prefix not in {"B", "I"}:
        raise ValueError(f"malformed BIO label: {label!r}")

    if not entity_type or entity_type != entity_type.strip():
        raise ValueError(f"malformed BIO label: {label!r}")

    return prefix, entity_type


def validate_label_mapping(
    label_to_id: Mapping[str, int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """Validate the mapping from BIO labels to model class IDs.

    Example:
        ``{"O": 0, "B-PERSON": 1, "I-PERSON": 2}`` is valid.
    """

    if not isinstance(label_to_id, Mapping):
        raise TypeError("label_to_id must be a mapping")

    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")

    ids: list[int] = []

    for label, label_id in label_to_id.items():
        parse_bio_label(label)

        if isinstance(label_id, bool) or not isinstance(label_id, int):
            raise TypeError(
                f"ID for label {label!r} must be an integer"
            )

        if label_id == ignore_index:
            raise ValueError(
                f"label ID for {label!r} must not equal ignore_index"
            )

        ids.append(label_id)

    if len(ids) != len(set(ids)):
        raise ValueError("label IDs must be unique")


def labels_to_ids(
    token_labels: Sequence[TokenLabel],
    label_to_id: Mapping[str, int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[int, ...]:
    """Convert token BIO labels into numeric model labels.

    ``None`` becomes ``ignore_index``.

    Example:
        ``("O", "B-PERSON", None)`` becomes ``(0, 1, -100)``.
    """

    validate_label_mapping(
        label_to_id,
        ignore_index=ignore_index,
    )

    converted: list[int] = []

    for token_index, label in enumerate(token_labels):
        if label is None:
            converted.append(ignore_index)
            continue

        parse_bio_label(label)

        if label not in label_to_id:
            raise ValueError(
                f"label {label!r} at token index {token_index} "
                "is missing from label_to_id"
            )

        converted.append(label_to_id[label])

    return tuple(converted)