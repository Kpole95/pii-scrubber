"""Tests for locating source words."""

import pytest

from pii_scrub.data.alignment import WordOffset, locate_words


def test_locate_words_maps_exact_offsets() -> None:
    """Ordinary words should map to their exact source positions."""

    offsets = locate_words(
        "Call Murali Krishna today.",
        ["Call", "Murali", "Krishna", "today", "."],
    )

    assert offsets == [
        WordOffset("Call", 0, 4),
        WordOffset("Murali", 5, 11),
        WordOffset("Krishna", 12, 19),
        WordOffset("today", 20, 25),
        WordOffset(".", 25, 26),
    ]


def test_locate_words_handles_repeated_words() -> None:
    """Repeated words should map to separate occurrences."""

    offsets = locate_words(
        "John called John.",
        ["John", "called", "John", "."],
    )

    assert offsets == [
        WordOffset("John", 0, 4),
        WordOffset("called", 5, 11),
        WordOffset("John", 12, 16),
        WordOffset(".", 16, 17),
    ]


def test_locate_words_preserves_irregular_whitespace() -> None:
    """Tabs, newlines, and repeated spaces must not change words."""

    text = "Call\tMurali\nKrishna  today."
    words = ["Call", "Murali", "Krishna", "today", "."]

    offsets = locate_words(text, words)

    assert [offset.extract(text) for offset in offsets] == words
    assert [(offset.start, offset.end) for offset in offsets] == [
        (0, 4),
        (5, 11),
        (12, 19),
        (21, 26),
        (26, 27),
    ]


def test_locate_words_handles_unicode() -> None:
    """Unicode offsets should use Python character positions."""

    offsets = locate_words(
        "Contact José Álvarez today.",
        ["Contact", "José", "Álvarez", "today", "."],
    )

    assert offsets[1] == WordOffset("José", 8, 12)
    assert offsets[2] == WordOffset("Álvarez", 13, 20)


def test_locate_words_returns_empty_list() -> None:
    """No annotated words should produce no offsets."""

    assert locate_words("No annotations.", []) == []


def test_locate_words_rejects_missing_word() -> None:
    """A missing source word must raise an error."""

    with pytest.raises(
        ValueError,
        match="word 'Krishna' at index 2 was not found",
    ):
        locate_words(
            "Call Murali today.",
            ["Call", "Murali", "Krishna", "today", "."],
        )


def test_locate_words_rejects_wrong_order() -> None:
    """Annotated words must follow source-text order."""

    with pytest.raises(ValueError):
        locate_words(
            "Murali called Krishna.",
            ["Krishna", "Murali", "called", "."],
        )