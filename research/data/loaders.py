"""Batch dataset loading with explicit rejection reports."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from research.data.gretel import load_gretel_finance_record
from research.data.models import DatasetExample, DatasetLoadReport, RejectedRecord
from research.data.openpii import load_openpii_record

RecordLoader = Callable[[Mapping[str, Any]], DatasetExample]


def load_records(
    records: Iterable[Mapping[str, Any]], *, loader: RecordLoader, strict: bool = False
) -> DatasetLoadReport:
    """Normalize many records and retain structured details for invalid rows.

    ``strict=True`` raises the first validation error. The default mode keeps
    valid rows and returns a rejection report suitable for dataset audits.
    """

    if isinstance(records, str):
        raise TypeError("records must be an iterable of mappings")
    if not callable(loader):
        raise TypeError("loader must be callable")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")

    accepted: list[DatasetExample] = []
    rejected: list[RejectedRecord] = []
    for index, record in enumerate(records):
        try:
            if not isinstance(record, Mapping):
                raise TypeError(f"record at index {index} must be a mapping")
            example = loader(record)
            if not isinstance(example, DatasetExample):
                raise TypeError("loader must return a DatasetExample")
        except (TypeError, ValueError) as error:
            if strict:
                raise
            rejected.append(RejectedRecord(index, type(error).__name__, str(error)))
        else:
            accepted.append(example)
    return DatasetLoadReport(tuple(accepted), tuple(rejected))


def load_openpii_records(
    records: Iterable[Mapping[str, Any]], *, strict: bool = False
) -> DatasetLoadReport:
    """Load a batch of OpenPII records."""

    return load_records(records, loader=load_openpii_record, strict=strict)


def load_gretel_finance_records(
    records: Iterable[Mapping[str, Any]], *, strict: bool = False
) -> DatasetLoadReport:
    """Load a batch of Gretel Finance records."""

    return load_records(records, loader=load_gretel_finance_record, strict=strict)


__all__ = [
    "load_gretel_finance_record",
    "load_gretel_finance_records",
    "load_openpii_record",
    "load_openpii_records",
    "load_records",
]
