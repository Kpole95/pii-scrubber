"""Tests for normalized dataset data models."""

from dataclasses import FrozenInstanceError

import pytest

from pii_scrub.text import CharacterSpan
from research.data.models import (
    DatasetExample,
    DatasetLoadReport,
    RejectedRecord,
)


def test_dataset_example_stores_normalized_record() -> None:
    """A valid example should preserve text, spans, and metadata."""

    example = DatasetExample(
        example_id="openpii-123",
        text="Call Murali at murali@example.com.",
        spans=(
            CharacterSpan(5, 11, "PERSON"),
            CharacterSpan(15, 33, "EMAIL"),
        ),
        source="openpii",
        language="en",
    )

    assert example.example_id == "openpii-123"
    assert example.text == "Call Murali at murali@example.com."
    assert example.source == "openpii"
    assert example.language == "en"
    assert example.entity_count == 2
    assert example.entity_types == frozenset(
        {
            "PERSON",
            "EMAIL",
        }
    )


def test_dataset_example_allows_no_entities() -> None:
    """A negative example may contain no annotated PII."""

    example = DatasetExample(
        example_id="negative-001",
        text="No personal information here.",
        spans=(),
        source="openpii",
        language="en",
    )

    assert example.entity_count == 0
    assert example.entity_types == frozenset()


@pytest.mark.parametrize(
    "example_id",
    [
        "",
        " ",
        "   ",
    ],
)
def test_dataset_example_rejects_empty_id(
    example_id: str,
) -> None:
    """Every record requires a useful identifier."""

    with pytest.raises(
        ValueError,
        match="example_id must not be empty",
    ):
        DatasetExample(
            example_id=example_id,
            text="Call Murali.",
            spans=(CharacterSpan(5, 11, "PERSON"),),
            source="openpii",
            language="en",
        )


def test_dataset_example_rejects_empty_text() -> None:
    """Training records must contain source text."""

    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        DatasetExample(
            example_id="example-001",
            text="",
            spans=(),
            source="openpii",
            language="en",
        )


def test_dataset_example_requires_tuple_spans() -> None:
    """Immutable examples should store spans as a tuple."""

    with pytest.raises(
        TypeError,
        match="spans must be a tuple",
    ):
        DatasetExample(
            example_id="example-001",
            text="Call Murali.",
            spans=[  # type: ignore[arg-type]
                CharacterSpan(5, 11, "PERSON"),
            ],
            source="openpii",
            language="en",
        )


def test_dataset_example_rejects_span_beyond_text() -> None:
    """A PII span cannot extend outside the source text."""

    with pytest.raises(
        ValueError,
        match="beyond text length",
    ):
        DatasetExample(
            example_id="example-001",
            text="Call Murali.",
            spans=(CharacterSpan(5, 20, "PERSON"),),
            source="openpii",
            language="en",
        )


def test_dataset_example_requires_sorted_spans() -> None:
    """Spans must follow their order in the original document."""

    with pytest.raises(
        ValueError,
        match="spans must be sorted",
    ):
        DatasetExample(
            example_id="example-001",
            text="Murali emailed murali@example.com.",
            spans=(
                CharacterSpan(15, 33, "EMAIL"),
                CharacterSpan(0, 6, "PERSON"),
            ),
            source="openpii",
            language="en",
        )


def test_dataset_example_rejects_overlapping_spans() -> None:
    """One text region cannot contain two overlapping gold entities."""

    with pytest.raises(
        ValueError,
        match="spans must not overlap",
    ):
        DatasetExample(
            example_id="example-001",
            text="Murali Krishna",
            spans=(
                CharacterSpan(0, 14, "PERSON"),
                CharacterSpan(7, 14, "SURNAME"),
            ),
            source="openpii",
            language="en",
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source", ""),
        ("source", " "),
        ("language", ""),
        ("language", " "),
    ],
)
def test_dataset_example_rejects_empty_metadata(
    field_name: str,
    value: str,
) -> None:
    """Dataset source and language must not be empty."""

    arguments = {
        "example_id": "example-001",
        "text": "Call Murali.",
        "spans": (CharacterSpan(5, 11, "PERSON"),),
        "source": "openpii",
        "language": "en",
    }

    arguments[field_name] = value

    with pytest.raises(ValueError):
        DatasetExample(**arguments)  # type: ignore[arg-type]


def test_dataset_example_is_immutable() -> None:
    """A validated dataset record must not change later."""

    example = DatasetExample(
        example_id="example-001",
        text="Call Murali.",
        spans=(CharacterSpan(5, 11, "PERSON"),),
        source="openpii",
        language="en",
    )

    with pytest.raises(FrozenInstanceError):
        example.text = "Changed"  # type: ignore[misc]


def test_rejected_record_stores_error_information() -> None:
    """A rejection should identify its record and error."""

    rejected = RejectedRecord(
        record_index=3,
        error_type="ValueError",
        message="unsupported entity label",
    )

    assert rejected.record_index == 3
    assert rejected.error_type == "ValueError"
    assert rejected.message == "unsupported entity label"


def test_dataset_load_report_calculates_counts() -> None:
    """The report should summarize accepted and rejected records."""

    example = DatasetExample(
        example_id="example-001",
        text="Call Murali.",
        spans=(CharacterSpan(5, 11, "PERSON"),),
        source="openpii",
        language="en",
    )

    report = DatasetLoadReport(
        examples=(example,),
        rejected=(
            RejectedRecord(
                record_index=1,
                error_type="ValueError",
                message="invalid annotation",
            ),
        ),
    )

    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.total_count == 2
    assert report.acceptance_rate == 0.5


def test_empty_dataset_load_report_has_zero_acceptance_rate() -> None:
    """An empty batch should avoid division by zero."""

    report = DatasetLoadReport(
        examples=(),
        rejected=(),
    )

    assert report.total_count == 0
    assert report.acceptance_rate == 0.0
