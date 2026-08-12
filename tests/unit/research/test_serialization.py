"""Tests for normalized dataset JSONL serialization."""

import json
from pathlib import Path

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
