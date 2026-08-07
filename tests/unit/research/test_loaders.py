"""Tests for shared dataset batch-loading behavior."""

from collections.abc import Mapping

import pytest

from research.data.loaders import (
    load_ai4privacy_records,
    load_conll2003_records,
    load_gretel_finance_records,
    load_openpii_record,
    load_openpii_records,
    load_records,
)


def _openpii(value: str = "John") -> dict[str, object]:
    """Return one minimal OpenPII row.

    Example:
        ``_openpii("Wrong")`` creates a deliberately mismatched annotation.
    """
    return {
        "source_text": "John",
        "uid": 1,
        "language": "en",
        "privacy_mask": [{"value": value, "start": 0, "end": 4, "label": "GIVENNAME"}],
    }


def test_openpii_batch_reports_bad_rows() -> None:
    """Batch loading should retain valid rows and report invalid ones."""
    report = load_openpii_records([_openpii(), _openpii("Wrong")])

    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.total_count == 2


def test_strict_mode_raises() -> None:
    """Strict batch loading should stop on the first invalid record."""
    with pytest.raises(ValueError, match="contains value"):
        load_openpii_records([_openpii("Wrong")], strict=True)


def test_ai4privacy_batch_uses_shared_loader() -> None:
    """Ai4Privacy convenience loading should report malformed rows."""
    rows = [
        {"id": 1, "language": "en", "source_text": "Hello", "privacy_mask": []},
        {"id": 2, "language": "en", "source_text": "Hello", "privacy_mask": "bad-json"},
    ]

    report = load_ai4privacy_records(rows)

    assert report.accepted_count == 1
    assert report.rejected_count == 1


def test_conll_batch_uses_shared_loader() -> None:
    """CoNLL convenience loading should report invalid tags."""
    report = load_conll2003_records(
        [
            {"tokens": ["John"], "tags": [3]},
            {"tokens": ["Mary"], "tags": [99]},
        ]
    )

    assert report.accepted_count == 1
    assert report.rejected_count == 1


def test_gretel_batch_uses_shared_loader() -> None:
    """Gretel convenience loading should normalize multiple rows."""
    rows = [
        {
            "index": 1,
            "language": "English",
            "generated_text": "Email john@example.com.",
            "pii_spans": [{"start": 6, "end": 22, "label": "EMAIL"}],
        },
        {
            "index": 2,
            "language": "English",
            "generated_text": "No PII.",
            "pii_spans": [],
        },
    ]

    report = load_gretel_finance_records(rows)

    assert report.accepted_count == 2
    assert report.rejected_count == 0


def test_rejects_non_mapping_batch_item() -> None:
    """Every batch item must be a raw-record mapping."""
    report = load_records(
        [_openpii(), "invalid"],  # type: ignore[list-item]
        loader=load_openpii_record,
    )

    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.rejected[0].error_type == "TypeError"


def test_rejects_invalid_loader_result() -> None:
    """A record loader must return DatasetExample objects."""

    def invalid_loader(record: Mapping[str, object]) -> str:
        return "invalid"

    report = load_records(
        [{"value": 1}],
        loader=invalid_loader,  # type: ignore[arg-type]
    )

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert report.rejected[0].message == "loader must return a DatasetExample"
