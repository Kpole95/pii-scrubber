"""Tests for Qwen dataset inspection helpers."""

import pytest

from scripts.check_qwen_dataset import _percentile


def test_percentile_minimum() -> None:
    """Zero percentile should return the smallest value."""

    assert _percentile([10, 20, 30], 0.0) == 10


def test_percentile_median() -> None:
    """Middle percentile should select the middle ordered value."""

    assert _percentile([10, 20, 30], 0.5) == 20


def test_percentile_maximum() -> None:
    """One hundredth percentile should return the largest value."""

    assert _percentile([10, 20, 30], 1.0) == 30


def test_percentile_rejects_empty_values() -> None:
    """Percentiles require at least one observation."""

    with pytest.raises(ValueError, match="must not be empty"):
        _percentile([], 0.5)


def test_percentile_rejects_invalid_fraction() -> None:
    """Percentile fraction must remain within zero and one."""

    with pytest.raises(ValueError, match="between 0 and 1"):
        _percentile([10], 1.1)
