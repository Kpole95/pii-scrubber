"""Tests for the canonical Ai4Privacy dataset adapter."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.ai4privacy import load_ai4privacy_record


def test_loads_supported_pii() -> None:
    """Supported labels should become canonical PII spans."""
    result = load_ai4privacy_record(
        {
            "id": 1,
            "language": "en",
            "source_text": "Call Murali at murali@example.com",
            "privacy_mask": [
                {"value": "Murali", "start": 5, "end": 11, "label": "FIRSTNAME"},
                {"value": "murali@example.com", "start": 15, "end": 33, "label": "EMAIL"},
            ],
        }
    )

    assert result.example_id == "ai4privacy-1"
    assert result.spans == (
        CharacterSpan(5, 11, "PERSON"),
        CharacterSpan(15, 33, "EMAIL"),
    )


def test_ignores_out_of_scope_label() -> None:
    """Out-of-scope labels should be validated but not returned."""
    result = load_ai4privacy_record(
        {
            "id": 2,
            "language": "en",
            "source_text": "Amount 500",
            "privacy_mask": [{"value": "500", "start": 7, "end": 10, "label": "AMOUNT"}],
        }
    )

    assert result.spans == ()


def test_accepts_json_privacy_mask() -> None:
    """Serialized privacy-mask JSON should also be accepted."""
    result = load_ai4privacy_record(
        {
            "id": 3,
            "language": "en",
            "source_text": "Email a@b.com",
            "privacy_mask": '[{"value":"a@b.com","start":6,"end":13,"label":"EMAIL"}]',
        }
    )

    assert result.spans == (CharacterSpan(6, 13, "EMAIL"),)


def test_rejects_bad_offsets() -> None:
    """Offsets must point to the exact annotated source text."""
    with pytest.raises(ValueError, match="does not match source text"):
        load_ai4privacy_record(
            {
                "id": 4,
                "language": "en",
                "source_text": "Call Murali",
                "privacy_mask": [{"value": "Krishna", "start": 5, "end": 11, "label": "FIRSTNAME"}],
            }
        )


def test_rejects_unknown_label() -> None:
    """Unknown upstream labels should fail instead of being silently ignored."""
    with pytest.raises(ValueError, match="unsupported Ai4Privacy label"):
        load_ai4privacy_record(
            {
                "id": 5,
                "language": "en",
                "source_text": "secret",
                "privacy_mask": [{"value": "secret", "start": 0, "end": 6, "label": "NEW_LABEL"}],
            }
        )
