"""Tests for the public research-data API."""

import research.data as data


def test_exports_stage4_adapters() -> None:
    """Stage 4 adapters should be available from ``research.data``.

    Example:
        ``research.data.load_ai4privacy_record`` is directly callable.
    """
    assert callable(data.load_ai4privacy_record)
    assert callable(data.load_ai4privacy_records)
    assert callable(data.load_conll2003_record)
    assert callable(data.load_conll2003_records)


def test_exports_existing_data_tools() -> None:
    """Existing dataset tools should remain public after Stage 4.

    Example:
        OpenPII loading and deterministic splitting remain accessible.
    """
    assert callable(data.load_openpii_record)
    assert callable(data.load_gretel_finance_record)
    assert callable(data.split_examples)
