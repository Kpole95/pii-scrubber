"""Official source labels and their normalized PII taxonomy."""

from collections.abc import Mapping, Set
from dataclasses import dataclass

OPENPII_LABELS: frozenset[str] = frozenset(
    {
        "AGE",
        "BUILDINGNUM",
        "CITY",
        "CREDITCARDNUMBER",
        "DATE",
        "DRIVERLICENSENUM",
        "EMAIL",
        "GENDER",
        "GIVENNAME",
        "IDCARDNUM",
        "PASSPORTNUM",
        "SEX",
        "SOCIALNUM",
        "STREET",
        "SURNAME",
        "TAXNUM",
        "TELEPHONENUM",
        "TITLE",
        "ZIPCODE",
    }
)
"""The 19 entity labels published by OpenPII."""


GRETEL_FINANCE_LABELS: frozenset[str] = frozenset(
    {
        "ACCOUNT_PIN",
        "API_KEY",
        "BANK_ROUTING_NUMBER",
        "BBAN",
        "COMPANY",
        "CREDIT_CARD_NUMBER",
        "CREDIT_CARD_SECURITY_CODE",
        "CUSTOMER_ID",
        "DATE",
        "DATE_OF_BIRTH",
        "DATE_TIME",
        "DRIVER_LICENSE_NUMBER",
        "EMAIL",
        "EMPLOYEE_ID",
        "FIRST_NAME",
        "IBAN",
        "IPV4",
        "IPV6",
        "LAST_NAME",
        "LOCAL_LATLNG",
        "NAME",
        "PASSPORT_NUMBER",
        "PASSWORD",
        "PHONE_NUMBER",
        "SSN",
        "STREET_ADDRESS",
        "SWIFT_BIC_CODE",
        "TIME",
        "USER_NAME",
    }
)
"""The 29 entity labels published by Gretel Finance."""


NORMALIZED_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "ADDRESS",
        "AGE",
        "BANK_ACCOUNT",
        "BANK_ROUTING_NUMBER",
        "CITY",
        "COMPANY",
        "CREDIT_CARD",
        "DATE",
        "DATE_TIME",
        "DRIVER_LICENSE",
        "EMAIL",
        "GENDER",
        "ID_NUMBER",
        "IP_ADDRESS",
        "LOCATION",
        "PASSPORT",
        "PASSWORD",
        "PERSON",
        "PHONE",
        "POSTAL_CODE",
        "SECRET",
        "SOCIAL_SECURITY_NUMBER",
        "SWIFT_CODE",
        "TAX_ID",
        "TIME",
        "TITLE",
        "USERNAME",
    }
)
"""Entity types used internally by the PII Scrubber."""


@dataclass(frozen=True, slots=True)
class LabelAudit:
    """Describe whether a source-label mapping is complete.

    Example:
        A mapping missing ``EMAIL`` returns it in ``missing_labels``.
    """

    missing_labels: frozenset[str]
    invalid_targets: frozenset[str]

    @property
    def is_complete(self) -> bool:
        """Return whether all labels have valid normalized targets."""

        return not self.missing_labels and not self.invalid_targets


def audit_label_mapping(
    source_labels: Set[str],
    label_mapping: Mapping[str, str],
) -> LabelAudit:
    """Check that every source label maps to a supported entity type.

    Example:
        ``{"EMAIL"}`` mapped to ``{"EMAIL": "EMAIL"}`` is complete.
    """

    missing_labels = frozenset(label for label in source_labels if label not in label_mapping)

    invalid_targets = frozenset(
        target
        for label, target in label_mapping.items()
        if label in source_labels and target not in NORMALIZED_ENTITY_TYPES
    )

    return LabelAudit(
        missing_labels=missing_labels,
        invalid_targets=invalid_targets,
    )
