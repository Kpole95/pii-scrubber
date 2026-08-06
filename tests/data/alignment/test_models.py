"""Tests for immutable alignment data models."""

from dataclasses import FrozenInstanceError

import pytest

from pii_scrub.data.alignment import (
    AlignedExample,
    CharacterSpan,
    WordOffset,
)


def test_character_span_extracts_entity() -> None:
    """A valid half-open span should extract its exact text."""

    span = CharacterSpan(5, 19, "PERSON")

    assert span.length == 14
    assert span.extract("Call Murali Krishna today.") == "Murali Krishna"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 5),
        (5, 5),
        (10, 5),
    ],
)
def test_character_span_rejects_invalid_boundaries(
    start: int,
    end: int,
) -> None:
    """Negative, empty, and reversed spans are invalid."""

    with pytest.raises(ValueError):
        CharacterSpan(start, end, "PERSON")


@pytest.mark.parametrize("entity_type", ["", " ", "   "])
def test_character_span_rejects_empty_entity_type(
    entity_type: str,
) -> None:
    """An entity type must contain visible characters."""

    with pytest.raises(ValueError):
        CharacterSpan(0, 4, entity_type)


def test_character_span_rejects_end_beyond_text() -> None:
    """Extraction must reject a span outside the source text."""

    span = CharacterSpan(5, 19, "PERSON")

    with pytest.raises(
        ValueError,
        match="span end 19 exceeds text length 12",
    ):
        span.extract("Call Murali.")


def test_character_span_is_immutable() -> None:
    """A validated entity span must not change later."""

    span = CharacterSpan(0, 4, "PERSON")

    with pytest.raises(FrozenInstanceError):
        span.start = 1  # type: ignore[misc]


def test_word_offset_extracts_word() -> None:
    """A word offset should extract and verify its source word."""

    offset = WordOffset("Murali", 5, 11)

    assert offset.extract("Call Murali.") == "Murali"


def test_word_offset_rejects_wrong_length() -> None:
    """The offset length must equal the stored word length."""

    with pytest.raises(ValueError):
        WordOffset("Murali", 5, 10)


def test_aligned_example_stores_token_fields() -> None:
    """All token-level fields should remain available."""

    example = AlignedExample(
        input_ids=(1, 4, 2),
        attention_mask=(1, 1, 1),
        offset_mapping=((0, 0), (0, 4), (0, 0)),
        word_ids=(None, 0, None),
        token_labels=(None, "O", None),
    )

    assert example.token_count == 3
    assert example.token_labels == (None, "O", None)


def test_aligned_example_rejects_different_lengths() -> None:
    """Every token-level tuple must have the same length."""

    with pytest.raises(
        ValueError,
        match="all AlignedExample fields must have equal lengths",
    ):
        AlignedExample(
            input_ids=(1, 4, 2),
            attention_mask=(1, 1),
            offset_mapping=((0, 0), (0, 4), (0, 0)),
            word_ids=(None, 0, None),
            token_labels=(None, "O", None),
        )


def test_aligned_example_rejects_empty_sequence() -> None:
    """An aligned example must contain tokenizer tokens."""

    with pytest.raises(
        ValueError,
        match="AlignedExample must contain at least one token",
    ):
        AlignedExample(
            input_ids=(),
            attention_mask=(),
            offset_mapping=(),
            word_ids=(),
            token_labels=(),
        )


def test_aligned_example_allows_ignored_source_subword() -> None:
    """A continuation subword may use label None."""

    example = AlignedExample(
        input_ids=(1, 5, 6, 2),
        attention_mask=(1, 1, 1, 1),
        offset_mapping=((0, 0), (5, 8), (8, 11), (0, 0)),
        word_ids=(None, 0, 0, None),
        token_labels=(None, "B-PERSON", None, None),
    )

    assert example.token_labels[2] is None


def test_aligned_example_requires_special_token_offset() -> None:
    """A token without a source word must use offset (0, 0)."""

    with pytest.raises(ValueError, match="must use offset"):
        AlignedExample(
            input_ids=(1,),
            attention_mask=(1,),
            offset_mapping=((0, 1),),
            word_ids=(None,),
            token_labels=(None,),
        )