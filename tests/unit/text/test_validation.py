"""Tests for BIO parsing and numeric-label conversion."""

import pytest

from pii_scrub.text import (
    labels_to_ids,
    parse_bio_label,
    validate_label_mapping,
)


def test_parse_bio_label_parses_o() -> None:
    """O represents text outside every entity."""

    assert parse_bio_label("O") == ("O", None)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("B-PERSON", ("B", "PERSON")),
        ("I-EMAIL", ("I", "EMAIL")),
        ("B-PHONE_NUMBER", ("B", "PHONE_NUMBER")),
        ("I-ID-NUMBER", ("I", "ID-NUMBER")),
    ],
)
def test_parse_bio_label_parses_entity(
    label: str,
    expected: tuple[str, str],
) -> None:
    """Valid B and I labels should return prefix and type."""

    assert parse_bio_label(label) == expected


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
def test_parse_bio_label_rejects_invalid_label(
    label: str,
) -> None:
    """Malformed BIO labels must fail immediately."""

    with pytest.raises(ValueError):
        parse_bio_label(label)


@pytest.mark.parametrize("label", [None, 12, True])
def test_parse_bio_label_rejects_non_string(
    label: object,
) -> None:
    """BIO labels must be strings."""

    with pytest.raises(TypeError):
        parse_bio_label(label)  # type: ignore[arg-type]


def test_validate_label_mapping_accepts_unique_ids() -> None:
    """Different labels may use different integer class IDs."""

    validate_label_mapping(
        {
            "O": 0,
            "B-PERSON": 1,
            "I-PERSON": 2,
        }
    )


def test_validate_label_mapping_rejects_duplicate_ids() -> None:
    """Different labels must not share one model class ID."""

    with pytest.raises(ValueError, match="label IDs must be unique"):
        validate_label_mapping(
            {
                "O": 0,
                "B-PERSON": 0,
            }
        )


def test_validate_label_mapping_reserves_ignore_index() -> None:
    """A real class must not use the ignored-token ID."""

    with pytest.raises(ValueError, match="must not equal ignore_index"):
        validate_label_mapping({"O": -100})


def test_labels_to_ids_converts_labels() -> None:
    """Strings become class IDs and None becomes -100."""

    result = labels_to_ids(
        token_labels=(
            None,
            "O",
            "B-PERSON",
            None,
            "I-PERSON",
        ),
        label_to_id={
            "O": 0,
            "B-PERSON": 1,
            "I-PERSON": 2,
        },
    )

    assert result == (-100, 0, 1, -100, 2)


def test_labels_to_ids_supports_custom_ignore_index() -> None:
    """The caller may select another ignored-token value."""

    assert labels_to_ids(
        token_labels=("O", None),
        label_to_id={"O": 0},
        ignore_index=-1,
    ) == (0, -1)


def test_labels_to_ids_rejects_missing_mapping() -> None:
    """Every real token label must exist in the mapping."""

    with pytest.raises(
        ValueError,
        match="label 'B-EMAIL' at token index 1 is missing",
    ):
        labels_to_ids(
            token_labels=("O", "B-EMAIL"),
            label_to_id={"O": 0},
        )
