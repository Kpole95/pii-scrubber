"""Parse generative Qwen PII predictions into detected spans."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pii_scrub.types import DetectedSpan


def parse_qwen_output(
    text: str,
    output: str,
    *,
    entity_types: set[str] | None = None,
) -> list[DetectedSpan]:
    """Parse one strict Qwen JSON response into document character spans.

    Example:
        ``{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}``
        becomes one ``DetectedSpan`` over ``[0, 4)``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(output, str):
        raise TypeError("output must be a string")

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen output must be valid JSON") from error

    if not isinstance(payload, Mapping):
        raise ValueError("Qwen output must be a JSON object")
    if set(payload) != {"spans"}:
        raise ValueError("Qwen output must contain only the 'spans' field")

    raw_spans = payload["spans"]
    if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, str):
        raise ValueError("'spans' must be a JSON array")

    spans = [
        _parse_span(
            item,
            text_length=len(text),
            entity_types=entity_types,
            index=index,
        )
        for index, item in enumerate(raw_spans)
    ]

    _validate_order(spans)
    return spans


def _parse_span(
    value: Any,
    *,
    text_length: int,
    entity_types: set[str] | None,
    index: int,
) -> DetectedSpan:
    """Parse and validate one generated span object."""

    if not isinstance(value, Mapping):
        raise ValueError(f"span at index {index} must be a JSON object")

    required = {"start", "end", "entity_type"}
    if set(value) != required:
        raise ValueError(f"span at index {index} must contain only start, end, and entity_type")

    start = value["start"]
    end = value["end"]
    entity_type = value["entity_type"]

    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise ValueError(f"span at index {index} offsets must be integers")

    if start < 0 or end <= start or end > text_length:
        raise ValueError(f"span at index {index} has invalid source offsets")

    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ValueError(f"span at index {index} must have a non-empty entity type")

    if entity_type != entity_type.strip():
        raise ValueError(f"span at index {index} entity type must not contain whitespace")

    if entity_types is not None and entity_type not in entity_types:
        raise ValueError(f"span at index {index} has unsupported entity type {entity_type!r}")

    return DetectedSpan(start, end, entity_type)


def _validate_order(spans: Sequence[DetectedSpan]) -> None:
    """Require generated spans to be sorted and non-overlapping."""

    previous: DetectedSpan | None = None

    for index, span in enumerate(spans):
        if previous is not None:
            if span.start < previous.start:
                raise ValueError("generated spans must be in ascending source order")
            if span.start < previous.end:
                raise ValueError(f"generated span at index {index} overlaps a previous span")

        previous = span
