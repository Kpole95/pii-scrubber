"""Load normalized dataset records with shared error reporting."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from research.data.ai4privacy import load_ai4privacy_record
from research.data.conll import load_conll2003_record
from research.data.gretel import load_gretel_finance_record
from research.data.models import DatasetExample, DatasetLoadReport, RejectedRecord
from research.data.openpii import load_openpii_record

RecordLoader = Callable[[Mapping[str, Any]], DatasetExample]


def load_records(
    records: Iterable[Mapping[str, Any]],
    *,
    loader: RecordLoader,
    strict: bool = False,
) -> DatasetLoadReport:
    """Load records and report invalid rows without stopping the batch.

    Example:
        A loader returning a string is recorded as a rejected row
        because loaders must return ``DatasetExample`` objects.
    """
    accepted: list[DatasetExample] = []
    rejected: list[RejectedRecord] = []

    for index, record in enumerate(records):
        try:
            if not isinstance(record, Mapping):
                raise TypeError("record must be a mapping")

            example = loader(record)

            if not isinstance(example, DatasetExample):
                raise TypeError("loader must return a DatasetExample")

            accepted.append(example)

        except (TypeError, ValueError) as error:
            if strict:
                raise

            rejected.append(
                RejectedRecord(
                    record_index=index,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

    return DatasetLoadReport(tuple(accepted), tuple(rejected))


def load_ai4privacy_records(
    records: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> DatasetLoadReport:
    """Load multiple Ai4Privacy rows.

    Example:
        ``load_ai4privacy_records(rows)`` reports malformed rows
        without discarding valid examples.
    """
    return load_records(records, loader=load_ai4privacy_record, strict=strict)


def load_conll2003_records(
    records: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> DatasetLoadReport:
    """Load multiple CoNLL-2003 rows.

    Example:
        A row containing tag ``99`` is recorded as rejected.
    """
    return load_records(records, loader=load_conll2003_record, strict=strict)


def load_openpii_records(
    records: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> DatasetLoadReport:
    """Load multiple OpenPII rows using the shared loader."""
    return load_records(records, loader=load_openpii_record, strict=strict)


def load_gretel_finance_records(
    records: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> DatasetLoadReport:
    """Load multiple Gretel Finance rows using the shared loader."""
    return load_records(records, loader=load_gretel_finance_record, strict=strict)
