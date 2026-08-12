"""Validation tests for normalized dataset JSONL serialization."""

import json
from pathlib import Path

import pytest

from pii_scrub.text import CharacterSpan
from research.data.models import DatasetExample
from research.data.serialization import (
    dataset_example_from_dict,
    dataset_example_to_dict,
    read_dataset_jsonl,
    write_dataset_jsonl,
)


def _make_example(
    *,
    example_id: str = "example-001",
) -> DatasetExample:
    """Create one normalized example for serialization tests."""

    return DatasetExample(
        example_id=example_id,
        text="Call José at jose@example.com.",
        spans=(
            CharacterSpan(5, 9, "PERSON"),
            CharacterSpan(13, 29, "EMAIL"),
        ),
        source="test",
        language="en",
    )


def test_read_dataset_jsonl_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON should report its line number."""

    valid_line = json.dumps(dataset_example_to_dict(_make_example()))

    path = tmp_path / "invalid.jsonl"
    path.write_text(
        f"{valid_line}\n{{invalid}}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"invalid JSON at line 2",
    ):
        read_dataset_jsonl(path)


def test_read_dataset_jsonl_rejects_empty_line(
    tmp_path: Path,
) -> None:
    """Blank lines should not be silently ignored."""

    valid_line = json.dumps(dataset_example_to_dict(_make_example()))

    path = tmp_path / "blank-line.jsonl"
    path.write_text(
        f"{valid_line}\n\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"empty line at line 2",
    ):
        read_dataset_jsonl(path)


def test_read_dataset_jsonl_reports_invalid_example_line(
    tmp_path: Path,
) -> None:
    """Invalid example fields should include the line number."""

    path = tmp_path / "invalid-example.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "example-001",
                "text": "John",
                "spans": [
                    {
                        "start": 0,
                        "end": 10,
                        "entity_type": "PERSON",
                    }
                ],
                "source": "test",
                "language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"invalid dataset example at line 1",
    ):
        read_dataset_jsonl(path)


def test_read_dataset_jsonl_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reading a missing file should raise FileNotFoundError."""

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        read_dataset_jsonl(tmp_path / "missing.jsonl")


def test_dataset_example_from_dict_requires_span_list() -> None:
    """The serialized spans field must be a JSON list."""

    data = dataset_example_to_dict(_make_example())
    data["spans"] = "invalid"

    with pytest.raises(
        TypeError,
        match="'spans' must be a list",
    ):
        dataset_example_from_dict(data)


def test_dataset_example_from_dict_rejects_invalid_span() -> None:
    """Every serialized span must contain valid fields."""

    data = dataset_example_to_dict(_make_example())
    data["spans"] = [
        {
            "start": 5,
            "end": 9,
        }
    ]

    with pytest.raises(
        ValueError,
        match="missing required field 'entity_type'",
    ):
        dataset_example_from_dict(data)


def test_write_dataset_jsonl_rejects_invalid_item(
    tmp_path: Path,
) -> None:
    """The writer should reject non-DatasetExample values."""

    with pytest.raises(
        TypeError,
        match=r"example at index 1 must be a DatasetExample",
    ):
        write_dataset_jsonl(
            [
                _make_example(),
                "invalid",  # type: ignore[list-item]
            ],
            tmp_path / "examples.jsonl",
        )
