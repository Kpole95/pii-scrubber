"""Normalize dataset-specific PII labels into one shared taxonomy."""

from collections.abc import Mapping

DEFAULT_LABEL_MAPPING: dict[str, str] = {
    # Names
    "PERSON": "PERSON",
    "NAME": "PERSON",
    "FULL_NAME": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    # Personal details
    "AGE": "AGE",
    "GENDER": "GENDER",
    "SEX": "GENDER",
    "TITLE": "TITLE",
    # Contact information
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TELEPHONE": "PHONE",
    "TELEPHONENUM": "PHONE",
    # Address and location
    "ADDRESS": "ADDRESS",
    "STREET": "ADDRESS",
    "STREET_ADDRESS": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "CITY": "CITY",
    "LOCATION": "LOCATION",
    "LOCAL_LATLNG": "LOCATION",
    "POSTAL_CODE": "POSTAL_CODE",
    "ZIP_CODE": "POSTAL_CODE",
    "ZIPCODE": "POSTAL_CODE",
    # Dates and times
    "DATE": "DATE",
    "DATE_OF_BIRTH": "DATE",
    "DOB": "DATE",
    "DATE_TIME": "DATE_TIME",
    "TIME": "TIME",
    # Identity documents
    "ID": "ID_NUMBER",
    "ID_NUMBER": "ID_NUMBER",
    "IDCARDNUM": "ID_NUMBER",
    "CUSTOMER_ID": "ID_NUMBER",
    "EMPLOYEE_ID": "ID_NUMBER",
    "PASSPORT": "PASSPORT",
    "PASSPORT_NUMBER": "PASSPORT",
    "PASSPORTNUM": "PASSPORT",
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    "DRIVERS_LICENSE": "DRIVER_LICENSE",
    "DRIVER_LICENSE_NUMBER": "DRIVER_LICENSE",
    "DRIVERLICENSENUM": "DRIVER_LICENSE",
    "TAX_ID": "TAX_ID",
    "TAXNUM": "TAX_ID",
    "SSN": "SOCIAL_SECURITY_NUMBER",
    "SOCIALNUM": "SOCIAL_SECURITY_NUMBER",
    # Financial information
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "ACCOUNT_NUMBER": "BANK_ACCOUNT",
    "BBAN": "BANK_ACCOUNT",
    "IBAN": "BANK_ACCOUNT",
    "BANK_ROUTING_NUMBER": "BANK_ROUTING_NUMBER",
    "SWIFT_BIC_CODE": "SWIFT_CODE",
    # Network and credentials
    "IP": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "USERNAME": "USERNAME",
    "USER_NAME": "USERNAME",
    "PASSWORD": "PASSWORD",
    "API_KEY": "SECRET",
    "ACCOUNT_PIN": "SECRET",
    "CREDIT_CARD_SECURITY_CODE": "SECRET",
    # Organization
    "COMPANY": "COMPANY",
    "ORGANIZATION": "COMPANY",
}


def normalize_entity_label(
    label: str,
    *,
    label_mapping: Mapping[str, str] = DEFAULT_LABEL_MAPPING,
) -> str:
    """Convert a dataset label into a normalized entity type.

    The function ignores capitalization and converts spaces and hyphens
    into underscores.

    Example:
        ``"first-name"`` becomes ``"PERSON"``.
        ``"email_address"`` becomes ``"EMAIL"``.
    """

    if not isinstance(label, str):
        raise TypeError("label must be a string")

    cleaned = label.strip()

    if not cleaned:
        raise ValueError("label must not be empty")

    normalized_key = cleaned.replace("-", "_").replace(" ", "_").upper()

    if normalized_key not in label_mapping:
        raise ValueError(f"unsupported entity label: {label!r}")

    normalized_label = label_mapping[normalized_key]

    if not isinstance(normalized_label, str):
        raise TypeError(f"normalized value for {normalized_key!r} must be a string")

    if not normalized_label.strip():
        raise ValueError(f"normalized value for {normalized_key!r} must not be empty")

    return normalized_label
