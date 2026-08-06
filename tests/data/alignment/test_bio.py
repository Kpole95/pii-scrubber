"""Tests for converting word BIO labels to character spans."""

import pytest

from pii_scrub.data.alignment import CharacterSpan, bio_tags_to_spans


def test_bio_tags_to_spans_builds_multiword_entity() -> None:
    """A B/I sequence should become one entity."""

    spans = bio_tags_to_spans(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
    )

    assert spans == [CharacterSpan(5, 19, "PERSON")]


def test_bio_tags_to_spans_builds_multiple_entities() -> None:
    """Separate B labels should create separate entities."""

    text = "Email Murali at murali@example.com."

    spans = bio_tags_to_spans(
        text=text,
        words=[
            "Email",
            "Murali",
            "at",
            "murali@example.com",
            ".",
        ],
        labels=[
            "O",
            "B-PERSON",
            "O",
            "B-EMAIL",
            "O",
        ],
    )

    assert spans == [
        CharacterSpan(6, 12, "PERSON"),
        CharacterSpan(16, 34, "EMAIL"),
    ]


def test_bio_tags_to_spans_closes_entity_at_end() -> None:
    """The final active entity must be stored."""

    assert bio_tags_to_spans(
        text="Contact Murali Krishna",
        words=["Contact", "Murali", "Krishna"],
        labels=["O", "B-PERSON", "I-PERSON"],
    ) == [CharacterSpan(8, 22, "PERSON")]


def test_bio_tags_to_spans_handles_adjacent_entities() -> None:
    """A new B label should immediately close the previous entity."""

    assert bio_tags_to_spans(
        text="Murali London",
        words=["Murali", "London"],
        labels=["B-PERSON", "B-LOCATION"],
    ) == [
        CharacterSpan(0, 6, "PERSON"),
        CharacterSpan(7, 13, "LOCATION"),
    ]


def test_bio_tags_to_spans_returns_empty_list_for_all_o() -> None:
    """An all-O sequence contains no PII entities."""

    assert bio_tags_to_spans(
        text="Nothing private here.",
        words=["Nothing", "private", "here", "."],
        labels=["O", "O", "O", "O"],
    ) == []


def test_bio_tags_to_spans_rejects_different_lengths() -> None:
    """Every source word requires exactly one BIO label."""

    with pytest.raises(
        ValueError,
        match="words and labels must contain the same number of items",
    ):
        bio_tags_to_spans(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "B-PERSON"],
        )


def test_bio_tags_to_spans_rejects_orphan_i() -> None:
    """I cannot appear without an active matching entity."""

    with pytest.raises(
        ValueError,
        match="I-label at index 1 has no active entity",
    ):
        bio_tags_to_spans(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "I-PERSON", "O"],
        )


def test_bio_tags_to_spans_rejects_type_change() -> None:
    """I must have the same type as the active entity."""

    with pytest.raises(
        ValueError,
        match="active entity has type 'PERSON'",
    ):
        bio_tags_to_spans(
            text="Murali Krishna",
            words=["Murali", "Krishna"],
            labels=["B-PERSON", "I-LOCATION"],
        )