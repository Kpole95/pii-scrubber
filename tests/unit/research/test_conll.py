"""Tests for the CoNLL-2003 PERSON-only dataset adapter."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.conll import load_conll2003_record


def test_loads_standard_huggingface_person_tags() -> None:
    """Standard Hugging Face B-PER/I-PER IDs should become PERSON."""

    result = load_conll2003_record(
        {
            "tokens": ["Peter", "Blackburn", "visited", "London", "."],
            "ner_tags": [1, 2, 0, 5, 0],
        }
    )

    assert result.text == "Peter Blackburn visited London."
    assert result.spans == (CharacterSpan(0, 15, "PERSON"),)


def test_standard_huggingface_org_tags_are_not_person() -> None:
    """Standard B-ORG/I-ORG IDs must not be interpreted as PERSON."""

    result = load_conll2003_record(
        {
            "tokens": ["European", "Commission", "in", "Brussels", "."],
            "ner_tags": [3, 4, 0, 5, 0],
        }
    )

    assert result.spans == ()


def test_supports_legacy_project_tag_ids() -> None:
    """Legacy project rows should retain their historical numeric mapping."""

    result = load_conll2003_record(
        {
            "tokens": ["Peter", "Blackburn"],
            "tags": [3, 4],
        }
    )

    assert result.spans == (CharacterSpan(0, 15, "PERSON"),)


def test_detokenizes_punctuation() -> None:
    """Punctuation should attach without unnecessary spaces."""

    result = load_conll2003_record(
        {
            "tokens": ["John", "'s", "team", "won", "."],
            "ner_tags": [1, 0, 0, 0, 0],
        }
    )

    assert result.text == "John's team won."
    assert result.spans == (CharacterSpan(0, 4, "PERSON"),)


def test_recovers_orphan_i_person() -> None:
    """A leading I-PER should start a recoverable PERSON span."""

    result = load_conll2003_record(
        {
            "tokens": ["John", "Smith"],
            "ner_tags": [2, 2],
        }
    )

    assert result.spans == (CharacterSpan(0, 10, "PERSON"),)


def test_rejects_unknown_tag() -> None:
    """Unknown tag IDs should fail instead of being silently ignored."""

    with pytest.raises(ValueError, match="unsupported CoNLL tag ID"):
        load_conll2003_record(
            {
                "tokens": ["John"],
                "ner_tags": [99],
            }
        )
