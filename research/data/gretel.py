"""Normalize records from the Gretel synthetic-finance PII schema."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.data.normalize import normalize_entity_label


def load_gretel_finance_record(record: Mapping[str, Any]) -> DatasetExample:
    """Convert one Gretel Finance record into a normalized example.

    ``pii_spans`` may be an already-decoded sequence or a JSON string.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    text = _required_text(record, "generated_text")
    language = _required_text(record, "language")
    index = _required_identifier(record, "index")
    raw_spans = _decode_spans(record)
    spans = tuple(
        sorted(
            (_parse_annotation(item, position, text) for position, item in enumerate(raw_spans)),
            key=lambda span: (span.start, span.end, span.entity_type),
        )
    )
    return DatasetExample(
        example_id=f"gretel-finance-{index}",
        text=text,
        spans=spans,
        source="gretel_finance",
        language=language,
    )


def _decode_spans(record: Mapping[str, Any]) -> Sequence[object]:
    """Decode the source span payload into Python data."""
    if "pii_spans" not in record:
        raise ValueError("Gretel record is missing required field 'pii_spans'")
    raw = record["pii_spans"]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("Gretel field 'pii_spans' contains invalid JSON") from error
    else:
        decoded = raw
    if not isinstance(decoded, Sequence) or isinstance(decoded, str):
        raise TypeError("Gretel field 'pii_spans' must decode to a sequence")
    return decoded


def _parse_annotation(item: object, index: int, text: str) -> CharacterSpan:
    """Parse one source annotation into a normalized span."""
    if not isinstance(item, Mapping):
        raise TypeError(f"Gretel annotation at index {index} must be a mapping")
    start = _required_int(item, "start", index)
    end = _required_int(item, "end", index)
    field = "label" if "label" in item else "type" if "type" in item else None
    if field is None:
        raise ValueError(f"Gretel annotation at index {index} is missing field 'label' or 'type'")
    label = _required_annotation_text(item, field, index)
    span = CharacterSpan(start, end, normalize_entity_label(label))
    if span.end > len(text):
        raise ValueError(
            f"Gretel annotation at index {index} ends at {span.end}, beyond text length {len(text)}"
        )
    span.extract(text)
    return span


def _required_text(record: Mapping[str, Any], field: str) -> str:
    """Read and validate one required text field."""
    if field not in record:
        raise ValueError(f"Gretel record is missing required field {field!r}")
    value = record[field]
    if not isinstance(value, str):
        raise TypeError(f"Gretel field {field!r} must be a string")
    if not value.strip():
        raise ValueError(f"Gretel field {field!r} must not be empty")
    return value


def _required_identifier(record: Mapping[str, Any], field: str) -> str:
    """Read and validate one required record identifier."""
    if field not in record:
        raise ValueError(f"Gretel record is missing required field {field!r}")
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError(f"Gretel field {field!r} must be a string or integer")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"Gretel field {field!r} must not be empty")
    return cleaned


def _required_annotation_text(item: Mapping[str, Any], field: str, index: int) -> str:
    """Read and validate one annotation text value."""
    value = item[field]
    if not isinstance(value, str):
        raise TypeError(f"Gretel annotation field {field!r} at index {index} must be a string")
    if not value.strip():
        raise ValueError(f"Gretel annotation field {field!r} at index {index} must not be empty")
    return value


def _required_int(item: Mapping[str, Any], field: str, index: int) -> int:
    """Read and validate one required integer field."""
    if field not in item:
        raise ValueError(f"Gretel annotation at index {index} is missing field {field!r}")
    value = item[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Gretel annotation field {field!r} at index {index} must be an integer")
    return value
