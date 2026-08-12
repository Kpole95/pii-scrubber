"""Tests for the OpenPII dataset adapter."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.data.openpii import load_openpii_record


def _record(**changes: object) -> dict[str, object]:
    """Return a valid OpenPII row with optional overrides.

    Example:
        ``_record(uid=7)`` returns the normal row with ID ``7``.
    """
    record: dict[str, object] = {
        "source_text": "John Smith can help.",
        "uid": 1,
        "language": "en",
        "privacy_mask": [
            {"value": "John", "start": 0, "end": 4, "label": "GIVENNAME"},
            {"value": "Smith", "start": 5, "end": 10, "label": "SURNAME"},
        ],
    }
    record.update(changes)
    return record


def test_normalizes_openpii_record() -> None:
    """A valid row should become one normalized DatasetExample."""
    assert load_openpii_record(_record()) == DatasetExample(
        example_id="openpii-1",
        text="John Smith can help.",
        spans=(CharacterSpan(0, 4, "PERSON"), CharacterSpan(5, 10, "PERSON")),
        source="openpii",
        language="en",
    )


def test_allows_string_id_and_empty_annotations() -> None:
    """String IDs and negative examples should remain valid."""
    result = load_openpii_record(
        _record(uid="abc", source_text="Nothing private.", privacy_mask=[])
    )

    assert result.example_id == "openpii-abc"
    assert result.spans == ()


def test_sorts_annotations() -> None:
    """Source annotation order should not affect normalized span order."""
    result = load_openpii_record(
        _record(
            source_text="John uses john@example.com.",
            privacy_mask=[
                {"value": "john@example.com", "start": 10, "end": 26, "label": "EMAIL"},
                {"value": "John", "start": 0, "end": 4, "label": "GIVENNAME"},
            ],
        )
    )

    assert result.spans == (
        CharacterSpan(0, 4, "PERSON"),
        CharacterSpan(10, 26, "EMAIL"),
    )


@pytest.mark.parametrize("field", ["source_text", "uid", "language"])
def test_rejects_missing_record_field(field: str) -> None:
    """Required OpenPII fields must be present."""
    record = _record()
    del record[field]

    with pytest.raises(ValueError, match="missing required field"):
        load_openpii_record(record)


@pytest.mark.parametrize("field", ["value", "start", "end", "label"])
def test_rejects_missing_annotation_field(field: str) -> None:
    """Every privacy-mask annotation requires its core fields."""
    annotation = {"value": "John", "start": 0, "end": 4, "label": "GIVENNAME"}
    del annotation[field]

    with pytest.raises(ValueError, match="is missing field"):
        load_openpii_record(_record(privacy_mask=[annotation]))


def test_rejects_invalid_annotation() -> None:
    """Annotation offsets and values must match the original text."""
    with pytest.raises(ValueError, match="contains value"):
        load_openpii_record(
            _record(privacy_mask=[{"value": "Jane", "start": 0, "end": 4, "label": "GIVENNAME"}])
        )


def test_rejects_unknown_label() -> None:
    """Unknown source labels must not silently enter the taxonomy."""
    with pytest.raises(ValueError, match="unsupported entity label"):
        load_openpii_record(
            _record(
                source_text="secret",
                privacy_mask=[
                    {
                        "value": "secret",
                        "start": 0,
                        "end": 6,
                        "label": "UNKNOWN_SECRET",
                    }
                ],
            )
        )
