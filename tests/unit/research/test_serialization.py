"""Tests for normalized dataset JSONL serialization."""

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


def test_dataset_example_to_dict_creates_json_data() -> None:
    """A normalized example should become plain JSON-compatible data."""

    result = dataset_example_to_dict(_make_example())

    assert result == {
        "example_id": "example-001",
        "text": "Call José at jose@example.com.",
        "spans": [
            {
                "start": 5,
                "end": 9,
                "entity_type": "PERSON",
            },
            {
                "start": 13,
                "end": 29,
                "entity_type": "EMAIL",
            },
        ],
        "source": "test",
        "language": "en",
    }


def test_dataset_example_from_dict_restores_example() -> None:
    """Decoded JSON data should recreate DatasetExample."""

    original = _make_example()

    restored = dataset_example_from_dict(dataset_example_to_dict(original))

    assert restored == original


def test_write_and_read_dataset_jsonl_round_trip(
    tmp_path: Path,
) -> None:
    """Written normalized examples should load without changes."""

    examples = (
        _make_example(example_id="example-001"),
        DatasetExample(
            example_id="example-002",
            text="Nothing private here.",
            spans=(),
            source="test",
            language="en",
        ),
    )

    path = tmp_path / "nested" / "examples.jsonl"

    written_count = write_dataset_jsonl(
        examples,
        path,
    )
    restored = read_dataset_jsonl(path)

    assert written_count == 2
    assert restored == examples


def test_write_dataset_jsonl_preserves_unicode(
    tmp_path: Path,
) -> None:
    """Unicode text should be written directly, not escaped."""

    path = tmp_path / "unicode.jsonl"

    write_dataset_jsonl(
        (_make_example(),),
        path,
    )

    content = path.read_text(encoding="utf-8")

    assert "José" in content
    assert "\\u00e9" not in content


def test_write_dataset_jsonl_uses_one_line_per_example(
    tmp_path: Path,
) -> None:
    """Each normalized example should occupy exactly one line."""

    path = tmp_path / "examples.jsonl"

    write_dataset_jsonl(
        (
            _make_example(example_id="example-001"),
            _make_example(example_id="example-002"),
        ),
        path,
    )

    lines = path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 2

    assert json.loads(lines[0])["example_id"] == ("example-001")
    assert json.loads(lines[1])["example_id"] == ("example-002")


def test_write_dataset_jsonl_allows_empty_dataset(
    tmp_path: Path,
) -> None:
    """Writing no examples should create an empty file."""

    path = tmp_path / "empty.jsonl"

    count = write_dataset_jsonl(
        (),
        path,
    )

    assert count == 0
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_read_dataset_jsonl_allows_empty_file(
    tmp_path: Path,
) -> None:
    """An empty JSONL file should load as an empty tuple."""

    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert read_dataset_jsonl(path) == ()


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
