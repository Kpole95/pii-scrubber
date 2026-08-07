"""Tests for the Gretel Finance dataset adapter."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.gretel import load_gretel_finance_record
from research.data.models import DatasetExample


def _record(**changes: object) -> dict[str, object]:
    """Return a valid Gretel row with optional overrides.

    Example:
        ``_record(index=7)`` returns the normal row with ID ``7``.
    """
    record: dict[str, object] = {
        "index": 1,
        "language": "English",
        "generated_text": "Account holder John Smith uses 12345678.",
        "pii_spans": (
            '[{"start":15,"end":25,"label":"name"},{"start":31,"end":39,"label":"account_number"}]'
        ),
    }
    record.update(changes)
    return record


def test_normalizes_gretel_record() -> None:
    """JSON spans should become one normalized DatasetExample."""
    assert load_gretel_finance_record(_record()) == DatasetExample(
        example_id="gretel-finance-1",
        text="Account holder John Smith uses 12345678.",
        spans=(
            CharacterSpan(15, 25, "PERSON"),
            CharacterSpan(31, 39, "BANK_ACCOUNT"),
        ),
        source="gretel_finance",
        language="English",
    )


def test_accepts_decoded_spans() -> None:
    """A library may provide pii_spans already decoded."""
    result = load_gretel_finance_record(
        _record(
            generated_text="Email john@example.com.",
            pii_spans=[
                {"start": 6, "end": 22, "type": "email_address"},
            ],
        )
    )

    assert result.spans == (CharacterSpan(6, 22, "EMAIL"),)


def test_allows_empty_spans() -> None:
    """A Gretel row may contain no PII annotations."""
    result = load_gretel_finance_record(
        _record(generated_text="General information.", pii_spans=[])
    )

    assert result.spans == ()


@pytest.mark.parametrize(
    "field",
    ["index", "language", "generated_text", "pii_spans"],
)
def test_rejects_missing_field(field: str) -> None:
    """Every required Gretel field must be present."""
    record = _record()
    del record[field]

    with pytest.raises(ValueError, match="missing required field"):
        load_gretel_finance_record(record)


def test_rejects_invalid_json() -> None:
    """Malformed serialized annotations should fail clearly."""
    with pytest.raises(ValueError, match="contains invalid JSON"):
        load_gretel_finance_record(_record(pii_spans="{bad json}"))


def test_rejects_missing_label() -> None:
    """Every annotation needs either a label or type."""
    with pytest.raises(ValueError, match="missing field 'label' or 'type'"):
        load_gretel_finance_record(
            _record(
                generated_text="John",
                pii_spans=[{"start": 0, "end": 4}],
            )
        )


def test_rejects_unknown_label() -> None:
    """Unknown Gretel labels must not enter the taxonomy."""
    with pytest.raises(ValueError, match="unsupported entity label"):
        load_gretel_finance_record(
            _record(
                generated_text="secret",
                pii_spans=[{"start": 0, "end": 6, "label": "unknown_financial_secret"}],
            )
        )
