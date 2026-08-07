"""Serialize normalized dataset examples as JSONL."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample


def dataset_example_to_dict(
    example: DatasetExample,
) -> dict[str, Any]:
    """Convert one normalized example into a JSON-compatible mapping.

    Example:
        A ``CharacterSpan`` becomes a dictionary containing its start,
        end, and normalized entity type.
    """

    if not isinstance(example, DatasetExample):
        raise TypeError("example must be a DatasetExample")

    return {
        "example_id": example.example_id,
        "text": example.text,
        "spans": [
            {
                "start": span.start,
                "end": span.end,
                "entity_type": span.entity_type,
            }
            for span in example.spans
        ],
        "source": example.source,
        "language": example.language,
    }


def dataset_example_from_dict(
    data: Mapping[str, Any],
) -> DatasetExample:
    """Create one normalized example from a decoded JSON object."""

    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")

    example_id = _require_string(
        data,
        field_name="example_id",
    )
    text = _require_string(
        data,
        field_name="text",
    )
    source = _require_string(
        data,
        field_name="source",
    )
    language = _require_string(
        data,
        field_name="language",
    )

    if "spans" not in data:
        raise ValueError("data is missing required field 'spans'")

    raw_spans = data["spans"]

    if not isinstance(raw_spans, list):
        raise TypeError("field 'spans' must be a list")

    spans = tuple(
        _character_span_from_dict(
            raw_span,
            span_index=span_index,
        )
        for span_index, raw_span in enumerate(raw_spans)
    )

    return DatasetExample(
        example_id=example_id,
        text=text,
        spans=spans,
        source=source,
        language=language,
    )


def write_dataset_jsonl(
    examples: Iterable[DatasetExample],
    path: str | Path,
) -> int:
    """Write normalized examples to a UTF-8 JSONL file.

    Returns:
        The number of examples written.
    """

    output_path = _normalize_path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for example_index, example in enumerate(examples):
            if not isinstance(example, DatasetExample):
                raise TypeError(f"example at index {example_index} must be a DatasetExample")

            serialized = dataset_example_to_dict(example)

            file.write(
                json.dumps(
                    serialized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

            count += 1

    return count


def read_dataset_jsonl(
    path: str | Path,
) -> tuple[DatasetExample, ...]:
    """Read normalized examples from a UTF-8 JSONL file."""

    input_path = _normalize_path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"dataset JSONL file does not exist: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"dataset JSONL path is not a file: {input_path}")

    examples: list[DatasetExample] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                raise ValueError(f"dataset JSONL contains an empty line at line {line_number}")

            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"dataset JSONL contains invalid JSON at line {line_number}"
                ) from error

            try:
                example = dataset_example_from_dict(decoded)
            except (TypeError, ValueError) as error:
                raise type(error)(
                    f"invalid dataset example at line {line_number}: {error}"
                ) from error

            examples.append(example)

    return tuple(examples)


def _character_span_from_dict(
    data: object,
    *,
    span_index: int,
) -> CharacterSpan:
    """Convert one decoded span dictionary into CharacterSpan."""

    if not isinstance(data, Mapping):
        raise TypeError(f"span at index {span_index} must be a mapping")

    start = _require_integer(
        data,
        field_name="start",
        context=f"span at index {span_index}",
    )
    end = _require_integer(
        data,
        field_name="end",
        context=f"span at index {span_index}",
    )
    entity_type = _require_string(
        data,
        field_name="entity_type",
        context=f"span at index {span_index}",
    )

    return CharacterSpan(
        start=start,
        end=end,
        entity_type=entity_type,
    )


def _require_string(
    data: Mapping[str, Any],
    *,
    field_name: str,
    context: str = "data",
) -> str:
    """Read one required non-empty string."""

    if field_name not in data:
        raise ValueError(f"{context} is missing required field {field_name!r}")

    value = data[field_name]

    if not isinstance(value, str):
        raise TypeError(f"{context} field {field_name!r} must be a string")

    if not value.strip():
        raise ValueError(f"{context} field {field_name!r} must not be empty")

    return value


def _require_integer(
    data: Mapping[str, Any],
    *,
    field_name: str,
    context: str,
) -> int:
    """Read one required integer."""

    if field_name not in data:
        raise ValueError(f"{context} is missing required field {field_name!r}")

    value = data[field_name]

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} field {field_name!r} must be an integer")

    return value


def _normalize_path(
    path: str | Path,
) -> Path:
    """Convert a supported path value into ``Path``."""

    if not isinstance(path, str | Path):
        raise TypeError("path must be a string or Path")

    normalized = Path(path)

    if not str(normalized).strip():
        raise ValueError("path must not be empty")

    return normalized
