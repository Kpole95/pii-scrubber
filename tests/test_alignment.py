import pytest

from pii_scrub.data.alignment import CharacterSpan


def test_half_open_character_span_extracts_exact_entity() -> None:
    """A [start, end) span must recover the exact original entity text."""

    text = "Call Murali Krishna today."

    person_start = text.index("Murali")
    person_end = person_start + len("Murali Krishna")

    assert person_start == 5
    assert person_end == 19
    assert text[person_start:person_end] == "Murali Krishna"


def test_character_span_stores_valid_entity_information() -> None:
    """A valid span should expose its boundaries, type, length and text."""

    text = "Call Murali Krishna today."
    span = CharacterSpan(
        start=5,
        end=19,
        entity_type="PERSON",
    )

    assert span.start == 5
    assert span.end == 19
    assert span.entity_type == "PERSON"
    assert span.length == 14
    assert span.extract(text) == "Murali Krishna"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 5),
        (5, 5),
        (19, 5),
    ],
)
def test_character_span_rejects_invalid_boundaries(
    start: int,
    end: int,
) -> None:
    """Negative, empty and reversed spans must be rejected."""

    with pytest.raises(ValueError):
        CharacterSpan(
            start=start,
            end=end,
            entity_type="PERSON",
        )


@pytest.mark.parametrize(
    "entity_type",
    [
        "",
        " ",
        "   ",
    ],
)
def test_character_span_rejects_empty_entity_type(
    entity_type: str,
) -> None:
    """Every span must contain a meaningful entity type."""

    with pytest.raises(ValueError):
        CharacterSpan(
            start=5,
            end=19,
            entity_type=entity_type,
        )


def test_character_span_rejects_end_beyond_text() -> None:
    """A span must not refer to characters beyond its source text."""

    text = "Call Murali."
    span = CharacterSpan(
        start=5,
        end=19,
        entity_type="PERSON",
    )

    with pytest.raises(
        ValueError,
        match="span end 19 exceeds text length 12",
    ):
        span.extract(text)