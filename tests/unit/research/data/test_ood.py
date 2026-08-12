"""Tests for the manually labeled OOD dataset adapter."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.ood import load_ood_record


def test_loads_ood_record() -> None:
    """A labeled row should become one normalized example."""
    result = load_ood_record(
        {
            "id": "001",
            "text": "Email Sarah at sarah@example.com",
            "spans": [
                {"start": 6, "end": 11, "label": "PERSON"},
                {"start": 15, "end": 32, "label": "EMAIL"},
            ],
        }
    )

    assert result.example_id == "ood-001"
    assert result.text == "Email Sarah at sarah@example.com"
    assert result.spans == (
        CharacterSpan(6, 11, "PERSON"),
        CharacterSpan(15, 32, "EMAIL"),
    )


def test_allows_negative_example() -> None:
    """An OOD example may intentionally contain no PII."""
    result = load_ood_record(
        {
            "id": "002",
            "text": "The meeting starts tomorrow.",
            "spans": [],
        }
    )

    assert result.spans == ()


def test_sorts_spans() -> None:
    """Manual annotation order should not affect output."""
    result = load_ood_record(
        {
            "id": "003",
            "text": "John uses john@example.com",
            "spans": [
                {"start": 10, "end": 26, "label": "EMAIL"},
                {"start": 0, "end": 4, "label": "PERSON"},
            ],
        }
    )

    assert result.spans == (
        CharacterSpan(0, 4, "PERSON"),
        CharacterSpan(10, 26, "EMAIL"),
    )


def test_rejects_span_beyond_text() -> None:
    """A manual span cannot extend beyond its source text."""
    with pytest.raises(ValueError, match="beyond text"):
        load_ood_record(
            {
                "id": "004",
                "text": "John",
                "spans": [{"start": 0, "end": 10, "label": "PERSON"}],
            }
        )


@pytest.mark.parametrize("field", ["id", "text"])
def test_rejects_missing_field(field: str) -> None:
    """ID and text are required for every OOD example."""
    record = {"id": "005", "text": "Hello", "spans": []}
    del record[field]

    with pytest.raises(ValueError, match="missing field"):
        load_ood_record(record)
