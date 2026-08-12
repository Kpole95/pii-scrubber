"""Tests for source-label taxonomy completeness."""

from research.data.normalize import (
    DEFAULT_LABEL_MAPPING,
)
from research.data.taxonomy import (
    GRETEL_FINANCE_LABELS,
    NORMALIZED_ENTITY_TYPES,
    OPENPII_LABELS,
    audit_label_mapping,
)


def test_openpii_contains_19_labels() -> None:
    """The OpenPII audit must cover its complete published taxonomy."""

    assert len(OPENPII_LABELS) == 19


def test_gretel_finance_contains_29_labels() -> None:
    """The Gretel audit must cover its complete published taxonomy."""

    assert len(GRETEL_FINANCE_LABELS) == 29


def test_default_mapping_covers_every_openpii_label() -> None:
    """Every OpenPII label must have an explicit normalized target."""

    audit = audit_label_mapping(
        OPENPII_LABELS,
        DEFAULT_LABEL_MAPPING,
    )

    assert audit.missing_labels == frozenset()
    assert audit.invalid_targets == frozenset()
    assert audit.is_complete


def test_default_mapping_covers_every_gretel_label() -> None:
    """Every Gretel Finance label must have a normalized target."""

    audit = audit_label_mapping(
        GRETEL_FINANCE_LABELS,
        DEFAULT_LABEL_MAPPING,
    )

    assert audit.missing_labels == frozenset()
    assert audit.invalid_targets == frozenset()
    assert audit.is_complete


def test_audit_reports_missing_label() -> None:
    """A source label without a mapping should be reported."""

    audit = audit_label_mapping(
        {"NAME", "EMAIL"},
        {
            "NAME": "PERSON",
        },
    )

    assert audit.missing_labels == frozenset({"EMAIL"})
    assert not audit.is_complete


def test_audit_reports_invalid_normalized_target() -> None:
    """Mappings cannot introduce unsupported internal entity types."""

    audit = audit_label_mapping(
        {"EMAIL"},
        {
            "EMAIL": "NOT_A_REAL_TYPE",
        },
    )

    assert audit.invalid_targets == frozenset({"NOT_A_REAL_TYPE"})
    assert not audit.is_complete


def test_every_default_target_is_uppercase() -> None:
    """Normalized entity names should use one consistent style."""

    assert all(target == target.upper() for target in DEFAULT_LABEL_MAPPING.values())


def test_expected_normalized_types_exist() -> None:
    """The taxonomy must contain core PII categories."""

    assert {
        "PERSON",
        "EMAIL",
        "PHONE",
        "ADDRESS",
        "ID_NUMBER",
        "BANK_ACCOUNT",
        "SECRET",
    }.issubset(NORMALIZED_ENTITY_TYPES)
