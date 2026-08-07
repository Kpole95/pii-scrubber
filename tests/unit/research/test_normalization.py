"""Tests for dataset entity-label normalization."""

import pytest

from research.data.normalize import (
    normalize_entity_label,
)


@pytest.mark.parametrize(
    "raw_label",
    [
        "PERSON",
        "person",
        "NAME",
        "full_name",
        "first_name",
        "first-name",
        "first name",
        "GIVENNAME",
        "surname",
        "last_name",
    ],
)
def test_normalize_entity_label_maps_person_labels(
    raw_label: str,
) -> None:
    """Different name labels should become PERSON."""

    assert normalize_entity_label(raw_label) == "PERSON"


@pytest.mark.parametrize(
    "raw_label",
    [
        "EMAIL",
        "email",
        "email_address",
        "email-address",
        "email address",
    ],
)
def test_normalize_entity_label_maps_email_labels(
    raw_label: str,
) -> None:
    """Different email labels should become EMAIL."""

    assert normalize_entity_label(raw_label) == "EMAIL"


@pytest.mark.parametrize(
    "raw_label",
    [
        "PHONE",
        "phone_number",
        "telephone",
        "TELEPHONENUM",
    ],
)
def test_normalize_entity_label_maps_phone_labels(
    raw_label: str,
) -> None:
    """Different telephone labels should become PHONE."""

    assert normalize_entity_label(raw_label) == "PHONE"


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("zip_code", "POSTAL_CODE"),
        ("postal-code", "POSTAL_CODE"),
        ("ip", "IP_ADDRESS"),
        ("ip_address", "IP_ADDRESS"),
        ("credit_card_number", "CREDIT_CARD"),
        ("account_number", "BANK_ACCOUNT"),
        ("passport_number", "PASSPORT"),
        ("drivers_license", "DRIVER_LICENSE"),
        ("date_of_birth", "DATE"),
        ("organization", "COMPANY"),
    ],
)
def test_normalize_entity_label_maps_common_labels(
    raw_label: str,
    expected: str,
) -> None:
    """Common dataset labels should map to shared types."""

    assert normalize_entity_label(raw_label) == expected


def test_normalize_entity_label_strips_whitespace() -> None:
    """Leading and trailing whitespace should be ignored."""

    assert normalize_entity_label("  email  ") == "EMAIL"


def test_normalize_entity_label_rejects_unknown_label() -> None:
    """Unknown labels must not silently become OTHER."""

    with pytest.raises(
        ValueError,
        match="unsupported entity label",
    ):
        normalize_entity_label("unknown_secret_type")


@pytest.mark.parametrize(
    "label",
    [
        "",
        " ",
        "   ",
    ],
)
def test_normalize_entity_label_rejects_empty_label(
    label: str,
) -> None:
    """Empty labels are invalid."""

    with pytest.raises(
        ValueError,
        match="label must not be empty",
    ):
        normalize_entity_label(label)


@pytest.mark.parametrize(
    "label",
    [
        None,
        123,
        True,
    ],
)
def test_normalize_entity_label_rejects_non_string(
    label: object,
) -> None:
    """Entity labels must be strings."""

    with pytest.raises(
        TypeError,
        match="label must be a string",
    ):
        normalize_entity_label(label)  # type: ignore[arg-type]


def test_normalize_entity_label_supports_custom_mapping() -> None:
    """A loader may provide an explicit dataset-specific mapping."""

    result = normalize_entity_label(
        "CUSTOM_NAME",
        label_mapping={
            "CUSTOM_NAME": "PERSON",
        },
    )

    assert result == "PERSON"
