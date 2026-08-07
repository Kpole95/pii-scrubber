"""Tests for deterministic replacement and exact restoration."""

import pytest

from pii_scrub.errors import InvalidSpanError, RestoreError
from pii_scrub.text.replacement import RestoreEntry, replace_spans, restore_text
from pii_scrub.types import CharacterSpan


def test_replace_and_restore_multiple_entity_types() -> None:
    text = "Email Ana at ana@example.com."
    result = replace_spans(
        text,
        [CharacterSpan(6, 9, "PERSON"), CharacterSpan(13, 28, "EMAIL")],
    )
    assert result.text == "Email [PERSON_1] at [EMAIL_1]."
    assert restore_text(result.text, result.mapping) == text


def test_repeated_values_get_distinct_placeholders() -> None:
    result = replace_spans(
        "Ana called Ana.",
        [CharacterSpan(0, 3, "PERSON"), CharacterSpan(11, 14, "PERSON")],
    )
    assert result.text == "[PERSON_1] called [PERSON_2]."


def test_replace_rejects_overlapping_spans() -> None:
    with pytest.raises(InvalidSpanError, match="must not overlap"):
        replace_spans(
            "abcdef",
            [CharacterSpan(0, 4, "PERSON"), CharacterSpan(3, 6, "EMAIL")],
        )


def test_restore_rejects_missing_placeholder() -> None:
    with pytest.raises(RestoreError, match="exactly once"):
        restore_text("unchanged", [RestoreEntry("[PERSON_1]", "Ana")])
