import pytest

from pii_scrub.data.alignment import CharacterSpan, parse_bio_label


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


def test_parse_bio_label_parses_outside_label() -> None:
    """The outside label should not contain an entity type."""

    prefix, entity_type = parse_bio_label("O")

    assert prefix == "O"
    assert entity_type is None


@pytest.mark.parametrize(
    ("label", "expected_prefix", "expected_entity_type"),
    [
        ("B-PERSON", "B", "PERSON"),
        ("I-EMAIL", "I", "EMAIL"),
        ("B-PHONE_NUMBER", "B", "PHONE_NUMBER"),
        ("I-ID-NUMBER", "I", "ID-NUMBER"),
    ],
)
def test_parse_bio_label_parses_entity_labels(
    label: str,
    expected_prefix: str,
    expected_entity_type: str,
) -> None:
    """Valid B and I labels should be split into prefix and entity type."""

    prefix, entity_type = parse_bio_label(label)

    assert prefix == expected_prefix
    assert entity_type == expected_entity_type


@pytest.mark.parametrize(
    "label",
    [
        "",
        "PERSON",
        "X-PERSON",
        "O-PERSON",
        "B-",
        "I-",
        " B-PERSON",
        "B-PERSON ",
        "B- PERSON",
    ],
)
def test_parse_bio_label_rejects_malformed_labels(
    label: str,
) -> None:
    """Malformed BIO labels must fail instead of being interpreted silently."""

    with pytest.raises(ValueError):
        parse_bio_label(label)


@pytest.mark.parametrize(
    "label",
    [
        None,
        12,
        True,
    ],
)
def test_parse_bio_label_rejects_non_string_values(
    label: object,
) -> None:
    """BIO labels must be strings."""

    with pytest.raises(TypeError):
        parse_bio_label(label)  # type: ignore[arg-type]