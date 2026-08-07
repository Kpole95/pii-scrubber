"""Tests for dataset-specific record loaders."""

from collections.abc import Mapping

import pytest

from pii_scrub.text import CharacterSpan
from research.data import DatasetExample
from research.data.loaders import (
    load_gretel_finance_record,
    load_gretel_finance_records,
    load_openpii_record,
    load_openpii_records,
    load_records,
)


def test_load_openpii_record_creates_dataset_example() -> None:
    """A valid OpenPII record should become one normalized example."""

    record = {
        "source_text": "John Smith can help.",
        "uid": 123,
        "language": "en",
        "privacy_mask": [
            {
                "value": "John",
                "start": 0,
                "end": 4,
                "label": "GIVENNAME",
            },
            {
                "value": "Smith",
                "start": 5,
                "end": 10,
                "label": "SURNAME",
            },
        ],
    }

    result = load_openpii_record(record)

    assert result == DatasetExample(
        example_id="openpii-123",
        text="John Smith can help.",
        spans=(
            CharacterSpan(0, 4, "PERSON"),
            CharacterSpan(5, 10, "PERSON"),
        ),
        source="openpii",
        language="en",
    )


def test_load_openpii_record_accepts_string_uid() -> None:
    """OpenPII identifiers may already be strings."""

    result = load_openpii_record(
        {
            "source_text": "Email me.",
            "uid": "abc-123",
            "language": "en",
            "privacy_mask": [],
        }
    )

    assert result.example_id == "openpii-abc-123"


def test_load_openpii_record_allows_no_annotations() -> None:
    """An OpenPII record may be a negative example."""

    result = load_openpii_record(
        {
            "source_text": "Nothing private here.",
            "uid": 10,
            "language": "en",
            "privacy_mask": [],
        }
    )

    assert result.spans == ()
    assert result.entity_count == 0


def test_load_openpii_record_sorts_annotations() -> None:
    """Raw annotation order should not affect the normalized example."""

    result = load_openpii_record(
        {
            "source_text": "John uses john@example.com.",
            "uid": 11,
            "language": "en",
            "privacy_mask": [
                {
                    "value": "john@example.com",
                    "start": 10,
                    "end": 26,
                    "label": "EMAIL",
                },
                {
                    "value": "John",
                    "start": 0,
                    "end": 4,
                    "label": "GIVENNAME",
                },
            ],
        }
    )

    assert result.spans == (
        CharacterSpan(0, 4, "PERSON"),
        CharacterSpan(10, 26, "EMAIL"),
    )


def test_load_openpii_record_rejects_value_mismatch() -> None:
    """The annotated value must match the exact source-text slice."""

    with pytest.raises(
        ValueError,
        match=r"contains value 'Jane'.*is 'John'",
    ):
        load_openpii_record(
            {
                "source_text": "John can help.",
                "uid": 12,
                "language": "en",
                "privacy_mask": [
                    {
                        "value": "Jane",
                        "start": 0,
                        "end": 4,
                        "label": "GIVENNAME",
                    }
                ],
            }
        )


def test_load_openpii_record_rejects_span_beyond_text() -> None:
    """An annotation must remain inside source_text."""

    with pytest.raises(
        ValueError,
        match="beyond text length",
    ):
        load_openpii_record(
            {
                "source_text": "John",
                "uid": 13,
                "language": "en",
                "privacy_mask": [
                    {
                        "value": "John",
                        "start": 0,
                        "end": 10,
                        "label": "GIVENNAME",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "source_text",
        "uid",
        "language",
    ],
)
def test_load_openpii_record_rejects_missing_record_field(
    missing_field: str,
) -> None:
    """Required top-level fields must be present."""

    record = {
        "source_text": "John can help.",
        "uid": 14,
        "language": "en",
        "privacy_mask": [],
    }

    del record[missing_field]

    with pytest.raises(ValueError, match="missing required field"):
        load_openpii_record(record)


def test_load_openpii_record_rejects_invalid_privacy_mask() -> None:
    """privacy_mask must contain a sequence of annotations."""

    with pytest.raises(
        TypeError,
        match="'privacy_mask' must be a sequence",
    ):
        load_openpii_record(
            {
                "source_text": "John can help.",
                "uid": 15,
                "language": "en",
                "privacy_mask": "invalid",
            }
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "value",
        "start",
        "end",
        "label",
    ],
)
def test_load_openpii_record_rejects_missing_annotation_field(
    missing_field: str,
) -> None:
    """Each privacy-mask annotation requires four fields."""

    annotation = {
        "value": "John",
        "start": 0,
        "end": 4,
        "label": "GIVENNAME",
    }

    del annotation[missing_field]

    with pytest.raises(ValueError, match="is missing field"):
        load_openpii_record(
            {
                "source_text": "John can help.",
                "uid": 16,
                "language": "en",
                "privacy_mask": [annotation],
            }
        )


def test_load_openpii_record_rejects_unknown_label() -> None:
    """Unknown labels must not silently enter the shared taxonomy."""

    with pytest.raises(
        ValueError,
        match="unsupported entity label",
    ):
        load_openpii_record(
            {
                "source_text": "secret",
                "uid": 17,
                "language": "en",
                "privacy_mask": [
                    {
                        "value": "secret",
                        "start": 0,
                        "end": 6,
                        "label": "UNKNOWN_SECRET",
                    }
                ],
            }
        )


def test_load_gretel_finance_record_decodes_json_spans() -> None:
    """A Gretel JSON span list should become a normalized example."""

    record = {
        "index": 42,
        "language": "English",
        "generated_text": ("Account holder John Smith uses 12345678."),
        "pii_spans": (
            "["
            '{"start": 15, "end": 25, "label": "name"},'
            '{"start": 31, "end": 39, '
            '"label": "account_number"}'
            "]"
        ),
    }

    result = load_gretel_finance_record(record)

    assert result == DatasetExample(
        example_id="gretel-finance-42",
        text="Account holder John Smith uses 12345678.",
        spans=(
            CharacterSpan(15, 25, "PERSON"),
            CharacterSpan(31, 39, "BANK_ACCOUNT"),
        ),
        source="gretel_finance",
        language="English",
    )


def test_load_gretel_finance_record_accepts_decoded_spans() -> None:
    """The loader may receive pii_spans already decoded by a library."""

    result = load_gretel_finance_record(
        {
            "index": 43,
            "language": "English",
            "generated_text": "Email john@example.com.",
            "pii_spans": [
                {
                    "start": 6,
                    "end": 22,
                    "type": "email_address",
                }
            ],
        }
    )

    assert result.spans == (CharacterSpan(6, 22, "EMAIL"),)


def test_load_gretel_finance_record_allows_no_spans() -> None:
    """A Gretel record may contain no annotated PII."""

    result = load_gretel_finance_record(
        {
            "index": 44,
            "language": "English",
            "generated_text": "General financial information.",
            "pii_spans": "[]",
        }
    )

    assert result.spans == ()
    assert result.entity_count == 0


def test_load_gretel_finance_record_sorts_spans() -> None:
    """Raw span order should not affect normalized output."""

    result = load_gretel_finance_record(
        {
            "index": 45,
            "language": "English",
            "generated_text": "John uses john@example.com.",
            "pii_spans": [
                {
                    "start": 10,
                    "end": 26,
                    "label": "email",
                },
                {
                    "start": 0,
                    "end": 4,
                    "label": "name",
                },
            ],
        }
    )

    assert result.spans == (
        CharacterSpan(0, 4, "PERSON"),
        CharacterSpan(10, 26, "EMAIL"),
    )


def test_load_gretel_finance_record_rejects_invalid_json() -> None:
    """Malformed pii_spans JSON must raise a clear error."""

    with pytest.raises(
        ValueError,
        match="'pii_spans' contains invalid JSON",
    ):
        load_gretel_finance_record(
            {
                "index": 46,
                "language": "English",
                "generated_text": "John can help.",
                "pii_spans": "{not valid JSON}",
            }
        )


def test_load_gretel_finance_record_rejects_non_list_json() -> None:
    """Decoded pii_spans must contain a sequence of annotations."""

    with pytest.raises(
        TypeError,
        match="'pii_spans' must decode to a sequence",
    ):
        load_gretel_finance_record(
            {
                "index": 47,
                "language": "English",
                "generated_text": "John can help.",
                "pii_spans": '{"start": 0, "end": 4}',
            }
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "index",
        "language",
        "generated_text",
        "pii_spans",
    ],
)
def test_load_gretel_finance_record_rejects_missing_field(
    missing_field: str,
) -> None:
    """Every required Gretel field must be present."""

    record = {
        "index": 48,
        "language": "English",
        "generated_text": "John can help.",
        "pii_spans": "[]",
    }

    del record[missing_field]

    with pytest.raises(ValueError, match="missing required field"):
        load_gretel_finance_record(record)


def test_load_gretel_finance_record_rejects_span_beyond_text() -> None:
    """A Gretel annotation cannot extend beyond generated_text."""

    with pytest.raises(
        ValueError,
        match="beyond text length",
    ):
        load_gretel_finance_record(
            {
                "index": 49,
                "language": "English",
                "generated_text": "John",
                "pii_spans": [
                    {
                        "start": 0,
                        "end": 10,
                        "label": "name",
                    }
                ],
            }
        )


def test_load_gretel_finance_record_rejects_missing_label() -> None:
    """Each Gretel annotation needs a label or type field."""

    with pytest.raises(
        ValueError,
        match="missing field 'label' or 'type'",
    ):
        load_gretel_finance_record(
            {
                "index": 50,
                "language": "English",
                "generated_text": "John",
                "pii_spans": [
                    {
                        "start": 0,
                        "end": 4,
                    }
                ],
            }
        )


def test_load_gretel_finance_record_rejects_unknown_label() -> None:
    """Unknown Gretel labels must not enter the taxonomy silently."""

    with pytest.raises(
        ValueError,
        match="unsupported entity label",
    ):
        load_gretel_finance_record(
            {
                "index": 51,
                "language": "English",
                "generated_text": "secret",
                "pii_spans": [
                    {
                        "start": 0,
                        "end": 6,
                        "label": "unknown_financial_secret",
                    }
                ],
            }
        )


def test_load_openpii_records_reports_valid_and_invalid_records() -> None:
    """Batch loading should keep valid rows and report bad rows."""

    records = [
        {
            "source_text": "John can help.",
            "uid": 1,
            "language": "en",
            "privacy_mask": [
                {
                    "value": "John",
                    "start": 0,
                    "end": 4,
                    "label": "GIVENNAME",
                }
            ],
        },
        {
            "source_text": "Jane can help.",
            "uid": 2,
            "language": "en",
            "privacy_mask": [
                {
                    "value": "Wrong",
                    "start": 0,
                    "end": 4,
                    "label": "GIVENNAME",
                }
            ],
        },
    ]

    report = load_openpii_records(records)

    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.total_count == 2

    assert report.examples[0].example_id == "openpii-1"
    assert report.rejected[0].record_index == 1
    assert report.rejected[0].error_type == "ValueError"
    assert "contains value" in report.rejected[0].message


def test_load_openpii_records_strict_mode_raises() -> None:
    """Strict loading should stop on the first invalid record."""

    records = [
        {
            "source_text": "John",
            "uid": 1,
            "language": "en",
            "privacy_mask": [
                {
                    "value": "Wrong",
                    "start": 0,
                    "end": 4,
                    "label": "GIVENNAME",
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="contains value"):
        load_openpii_records(
            records,
            strict=True,
        )


def test_load_gretel_finance_records_loads_batch() -> None:
    """The Gretel convenience loader should normalize many rows."""

    records = [
        {
            "index": 1,
            "language": "English",
            "generated_text": "Email john@example.com.",
            "pii_spans": [
                {
                    "start": 6,
                    "end": 22,
                    "label": "EMAIL",
                }
            ],
        },
        {
            "index": 2,
            "language": "English",
            "generated_text": "No PII here.",
            "pii_spans": [],
        },
    ]

    report = load_gretel_finance_records(records)

    assert report.accepted_count == 2
    assert report.rejected_count == 0


def test_load_records_rejects_non_mapping_item() -> None:
    """A batch item must be a raw record mapping."""

    report = load_records(
        [
            {
                "source_text": "No PII.",
                "uid": 1,
                "language": "en",
                "privacy_mask": [],
            },
            "invalid",  # type: ignore[list-item]
        ],
        loader=load_openpii_record,
    )

    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.rejected[0].record_index == 1
    assert report.rejected[0].error_type == "TypeError"


def test_load_records_rejects_loader_with_wrong_return_type() -> None:
    """A loader must return DatasetExample objects."""

    def invalid_loader(
        record: Mapping[str, object],
    ) -> str:
        return "invalid"

    report = load_records(
        [{"value": 1}],
        loader=invalid_loader,  # type: ignore[arg-type]
    )

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert report.rejected[0].message == ("loader must return a DatasetExample")
