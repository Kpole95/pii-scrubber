"""Tests for reconstructing spans from token BIO labels."""

import pytest

from pii_scrub.text import (
    AlignedExample,
    CharacterSpan,
    align_bio_to_subwords,
    aligned_labels_to_spans,
)
from tests.unit.text.conftest import FakeFastTokenizer


def test_reconstruction_builds_multiple_entities(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Separate B labels should reconstruct separate entities."""

    aligned = align_bio_to_subwords(
        text="Murali Krishna",
        words=["Murali", "Krishna"],
        labels=["B-PERSON", "B-LOCATION"],
        tokenizer=toy_tokenizer,
    )

    assert aligned_labels_to_spans(aligned) == [
        CharacterSpan(0, 6, "PERSON"),
        CharacterSpan(7, 14, "LOCATION"),
    ]


def test_reconstruction_returns_empty_for_all_o(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """An all-O sequence should reconstruct no PII spans."""

    aligned = align_bio_to_subwords(
        text="Call today.",
        words=["Call", "today", "."],
        labels=["O", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert aligned_labels_to_spans(aligned) == []


def test_reconstruction_rejects_orphan_i() -> None:
    """I cannot appear without an active entity."""

    example = AlignedExample(
        input_ids=(1, 5, 2),
        attention_mask=(1, 1, 1),
        offset_mapping=((0, 0), (0, 3), (0, 0)),
        word_ids=(None, 0, None),
        token_labels=(None, "I-PERSON", None),
    )

    with pytest.raises(
        ValueError,
        match="I-label at token index 1 has no active entity",
    ):
        aligned_labels_to_spans(example)


def test_reconstruction_rejects_type_change() -> None:
    """I must match the active entity type."""

    example = AlignedExample(
        input_ids=(1, 5, 7, 2),
        attention_mask=(1, 1, 1, 1),
        offset_mapping=((0, 0), (0, 3), (4, 11), (0, 0)),
        word_ids=(None, 0, 1, None),
        token_labels=(
            None,
            "B-PERSON",
            "I-LOCATION",
            None,
        ),
    )

    with pytest.raises(
        ValueError,
        match="active entity has type 'PERSON'",
    ):
        aligned_labels_to_spans(example)


def test_ignored_continuation_extends_entity() -> None:
    """A first-subword entity should include its ignored continuation."""

    example = AlignedExample(
        input_ids=(1, 5, 6, 2),
        attention_mask=(1, 1, 1, 1),
        offset_mapping=((0, 0), (5, 8), (8, 11), (0, 0)),
        word_ids=(None, 0, 0, None),
        token_labels=(None, "B-PERSON", None, None),
    )

    assert aligned_labels_to_spans(example) == [
        CharacterSpan(5, 11, "PERSON"),
    ]


def test_special_token_does_not_extend_entity() -> None:
    """A special token with word_id None must not change the span."""

    example = AlignedExample(
        input_ids=(1, 5, 2),
        attention_mask=(1, 1, 1),
        offset_mapping=((0, 0), (5, 11), (0, 0)),
        word_ids=(None, 0, None),
        token_labels=(None, "B-PERSON", None),
    )

    assert aligned_labels_to_spans(example) == [
        CharacterSpan(5, 11, "PERSON"),
    ]
