"""Tests for the CoNLL-2003 PERSON-only dataset adapter."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.conll import load_conll2003_record


def test_loads_multi_token_person() -> None:
    """B-PER/I-PER should become one PERSON character span."""
    result = load_conll2003_record(
        {
            "tokens": ["Peter", "Blackburn", "visited", "London", "."],
            "tags": [3, 4, 0, 5, 0],
        }
    )

    assert result.text == "Peter Blackburn visited London."
    assert result.spans == (CharacterSpan(0, 15, "PERSON"),)


def test_ignores_non_person_entities() -> None:
    """ORG and LOC tags should not become PII spans."""
    result = load_conll2003_record(
        {
            "tokens": ["European", "Commission", "in", "Brussels", "."],
            "tags": [1, 6, 0, 5, 0],
        }
    )

    assert result.spans == ()


def test_detokenizes_punctuation() -> None:
    """Punctuation should attach without unnecessary spaces."""
    result = load_conll2003_record(
        {
            "tokens": ["John", "'s", "team", "won", "."],
            "tags": [3, 0, 0, 0, 0],
        }
    )

    assert result.text == "John's team won."
    assert result.spans == (CharacterSpan(0, 4, "PERSON"),)


def test_recovers_orphan_i_person() -> None:
    """A leading I-PER should start a recoverable PERSON span."""
    result = load_conll2003_record(
        {
            "tokens": ["John", "Smith"],
            "tags": [4, 4],
        }
    )

    assert result.spans == (CharacterSpan(0, 10, "PERSON"),)


def test_rejects_unknown_tag() -> None:
    """Unknown tag IDs should fail instead of being silently ignored."""
    with pytest.raises(ValueError, match="unsupported CoNLL tag ID"):
        load_conll2003_record(
            {
                "tokens": ["John"],
                "tags": [99],
            }
        )
