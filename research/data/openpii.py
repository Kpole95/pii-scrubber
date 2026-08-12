"""Normalize records from the OpenPII-style privacy-mask schema."""

from collections.abc import Mapping, Sequence
from typing import Any

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.data.normalize import normalize_entity_label


def load_openpii_record(record: Mapping[str, Any]) -> DatasetExample:
    """Convert one raw OpenPII record into a validated normalized example.

    Annotation values must exactly equal ``source_text[start:end]`` so offset
    errors are detected before training or evaluation.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    text = _required_text(record, "source_text")
    language = _required_text(record, "language")
    uid = _required_identifier(record, "uid")
    annotations = record.get("privacy_mask")
    if not isinstance(annotations, Sequence) or isinstance(annotations, str):
        raise TypeError("OpenPII field 'privacy_mask' must be a sequence")

    spans = tuple(
        sorted(
            (_parse_annotation(item, index, text) for index, item in enumerate(annotations)),
            key=lambda span: (span.start, span.end, span.entity_type),
        )
    )
    return DatasetExample(
        example_id=f"openpii-{uid}",
        text=text,
        spans=spans,
        source="openpii",
        language=language,
    )


def _parse_annotation(item: object, index: int, text: str) -> CharacterSpan:
    """Parse one source annotation into a normalized span."""
    if not isinstance(item, Mapping):
        raise TypeError(f"OpenPII annotation at index {index} must be a mapping")
    start = _required_int(item, "start", index)
    end = _required_int(item, "end", index)
    label = _required_annotation_text(item, "label", index)
    value = _required_annotation_text(item, "value", index)
    span = CharacterSpan(start, end, normalize_entity_label(label))
    if span.end > len(text):
        raise ValueError(
            "OpenPII annotation at "
            f"index {index} ends at {span.end}, beyond text length {len(text)}"
        )
    extracted = span.extract(text)
    if extracted != value:
        raise ValueError(
            f"OpenPII annotation at index {index} contains value {value!r}, "
            f"but text[{start}:{end}] is {extracted!r}"
        )
    return span


def _required_text(record: Mapping[str, Any], field: str) -> str:
    """Read and validate one required text field."""
    if field not in record:
        raise ValueError(f"OpenPII record is missing required field {field!r}")
    value = record[field]
    if not isinstance(value, str):
        raise TypeError(f"OpenPII field {field!r} must be a string")
    if not value.strip():
        raise ValueError(f"OpenPII field {field!r} must not be empty")
    return value


def _required_identifier(record: Mapping[str, Any], field: str) -> str:
    """Read and validate one required record identifier."""
    if field not in record:
        raise ValueError(f"OpenPII record is missing required field {field!r}")
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError(f"OpenPII field {field!r} must be a string or integer")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"OpenPII field {field!r} must not be empty")
    return cleaned


def _required_annotation_text(item: Mapping[str, Any], field: str, index: int) -> str:
    """Read and validate one annotation text value."""
    if field not in item:
        raise ValueError(f"OpenPII annotation at index {index} is missing field {field!r}")
    value = item[field]
    if not isinstance(value, str):
        raise TypeError(f"OpenPII annotation field {field!r} at index {index} must be a string")
    if not value.strip():
        raise ValueError(f"OpenPII annotation field {field!r} at index {index} must not be empty")
    return value


def _required_int(item: Mapping[str, Any], field: str, index: int) -> int:
    """Read and validate one required integer field."""
    if field not in item:
        raise ValueError(f"OpenPII annotation at index {index} is missing field {field!r}")
    value = item[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"OpenPII annotation field {field!r} at index {index} must be an integer")
    return value
