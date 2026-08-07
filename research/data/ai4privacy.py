"""Convert Ai4Privacy records into the project's normalized dataset format."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample

SOURCE: Final = "ai4privacy_pii_masking_200k"

LABEL_MAP: Final = {
    "ACCOUNTNUMBER": "BANK_ACCOUNT",
    "BIC": "SWIFT_CODE",
    "BUILDINGNUMBER": "ADDRESS",
    "COMPANYNAME": "COMPANY",
    "CREDITCARDCVV": "SECRET",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "DOB": "DATE",
    "EMAIL": "EMAIL",
    "FIRSTNAME": "PERSON",
    "IBAN": "BANK_ACCOUNT",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "JOBTITLE": "TITLE",
    "LASTNAME": "PERSON",
    "MASKEDNUMBER": "ID_NUMBER",
    "MIDDLENAME": "PERSON",
    "NEARBYGPSCOORDINATE": "LOCATION",
    "PASSWORD": "PASSWORD",
    "PHONENUMBER": "PHONE",
    "PHONEIMEI": "ID_NUMBER",
    "PIN": "SECRET",
    "SECONDARYADDRESS": "ADDRESS",
    "SSN": "SOCIAL_SECURITY_NUMBER",
    "STREET": "ADDRESS",
    "USERNAME": "USERNAME",
    "VEHICLEVIN": "ID_NUMBER",
    "VEHICLEVRM": "ID_NUMBER",
    "ZIPCODE": "POSTAL_CODE",
}

IGNORED_LABELS: Final = {
    "ACCOUNTNAME",
    "AGE",
    "AMOUNT",
    "BITCOINADDRESS",
    "CITY",
    "COUNTY",
    "CREDITCARDISSUER",
    "CURRENCY",
    "CURRENCYCODE",
    "CURRENCYNAME",
    "CURRENCYSYMBOL",
    "DATE",
    "ETHEREUMADDRESS",
    "EYECOLOR",
    "GENDER",
    "HEIGHT",
    "JOBAREA",
    "JOBTYPE",
    "LITECOINADDRESS",
    "MAC",
    "ORDINALDIRECTION",
    "PREFIX",
    "SEX",
    "STATE",
    "TIME",
    "URL",
    "USERAGENT",
}

KNOWN_LABELS: Final = set(LABEL_MAP) | IGNORED_LABELS


def load_ai4privacy_record(record: Mapping[str, Any]) -> DatasetExample:
    """Convert one Ai4Privacy row into a normalized dataset example.

    Example:
        ``FIRSTNAME: Murali`` at ``[5, 11)`` becomes a ``PERSON`` span.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    text = _required(record, "source_text", str)
    language = _required(record, "language", str)
    example_id = str(_required(record, "id", (str, int))).strip()
    mask = record.get("privacy_mask")

    if isinstance(mask, str):
        try:
            mask = json.loads(mask)
        except json.JSONDecodeError as error:
            raise ValueError("privacy_mask contains invalid JSON") from error

    if not isinstance(mask, Sequence) or isinstance(mask, str):
        raise TypeError("privacy_mask must be a sequence")

    spans = tuple(
        span
        for index, item in enumerate(mask)
        if (span := _parse_span(item, index, text)) is not None
    )

    return DatasetExample(
        example_id=f"ai4privacy-{example_id}",
        text=text,
        spans=tuple(sorted(spans, key=lambda span: (span.start, span.end))),
        source=SOURCE,
        language=language,
    )


def _parse_span(item: object, index: int, text: str) -> CharacterSpan | None:
    """Validate and normalize one Ai4Privacy annotation.

    Example:
        ``EMAIL`` returns an EMAIL span; ``AMOUNT`` returns ``None``.
    """
    if not isinstance(item, Mapping):
        raise TypeError(f"annotation {index} must be a mapping")

    start = _required(item, "start", int)
    end = _required(item, "end", int)
    value = _required(item, "value", str)
    label = _required(item, "label", str).upper()

    if label not in KNOWN_LABELS:
        raise ValueError(f"unsupported Ai4Privacy label: {label!r}")

    span = CharacterSpan(start, end, label)

    if end > len(text) or span.extract(text) != value:
        raise ValueError(f"annotation {index} does not match source text")

    entity = LABEL_MAP.get(label)
    return CharacterSpan(start, end, entity) if entity else None


def _required(
    data: Mapping[str, Any],
    field: str,
    expected: type | tuple[type, ...],
) -> Any:
    """Return a required field with its expected type.

    Example:
        ``_required(row, "language", str)`` returns ``"en"``.
    """
    if field not in data:
        raise ValueError(f"missing field {field!r}")

    value = data[field]

    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(f"{field!r} has invalid type")

    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field!r} must not be empty")

    return value
