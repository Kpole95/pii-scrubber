"""Load manually labeled out-of-distribution examples."""

from collections.abc import Mapping, Sequence
from typing import Any

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample

SOURCE = "hand_labeled_ood"


def load_ood_record(record: Mapping[str, Any]) -> DatasetExample:
    """Convert one hand-labeled row into a DatasetExample.

    Example:
        ``{"id": "001", "text": "Hi John", ...}`` becomes
        example ``ood-001``.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    example_id = _required(record, "id", str)
    text = _required(record, "text", str)
    raw_spans = record.get("spans", [])

    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, str):
        raise TypeError("spans must be a sequence")

    spans = tuple(_span(span, text) for span in raw_spans)

    return DatasetExample(
        example_id=f"ood-{example_id}",
        text=text,
        spans=tuple(sorted(spans, key=lambda span: (span.start, span.end))),
        source=SOURCE,
        language="en",
    )


def _span(data: object, text: str) -> CharacterSpan:
    """Validate one manually labeled character span.

    Example:
        ``{"start": 3, "end": 7, "label": "PERSON"}`` becomes
        ``CharacterSpan(3, 7, "PERSON")``.
    """
    if not isinstance(data, Mapping):
        raise TypeError("span must be a mapping")

    start = _required(data, "start", int)
    end = _required(data, "end", int)
    label = _required(data, "label", str)
    span = CharacterSpan(start, end, label)

    if end > len(text):
        raise ValueError("span extends beyond text")

    return span


def _required(
    data: Mapping[str, Any],
    field: str,
    expected: type,
) -> Any:
    """Return one required field after basic validation.

    Example:
        ``_required(row, "id", str)`` returns the string ID.
    """
    if field not in data:
        raise ValueError(f"missing field {field!r}")

    value = data[field]

    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(f"{field!r} has invalid type")

    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field!r} must not be empty")

    return value
