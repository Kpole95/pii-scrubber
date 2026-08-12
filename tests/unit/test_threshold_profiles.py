"""Tests for frozen runtime threshold profiles."""

import json
from pathlib import Path

import pytest

from pii_scrub.threshold_profiles import load_threshold_profile


def _write(path: Path, data: object) -> None:
    """Write one small threshold configuration for a test."""

    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_balanced_profile(tmp_path: Path) -> None:
    """The balanced threshold profile should load by name."""

    path = tmp_path / "thresholds.yaml"
    _write(
        path,
        {
            "balanced": {"PERSON": 0.6},
            "strict": {"PERSON": 0.35},
        },
    )

    assert load_threshold_profile("balanced", path=path) == {"PERSON": 0.6}


def test_load_strict_profile(tmp_path: Path) -> None:
    """The strict threshold profile should load by name."""

    path = tmp_path / "thresholds.yaml"
    _write(
        path,
        {
            "balanced": {},
            "strict": {"EMAIL": 0.2},
        },
    )

    assert load_threshold_profile("strict", path=path) == {"EMAIL": 0.2}


def test_unknown_mode_fails(tmp_path: Path) -> None:
    """An unknown recall mode should fail before reading a config file."""

    with pytest.raises(ValueError, match="recall_mode"):
        load_threshold_profile("unknown", path=tmp_path / "unused")


def test_invalid_threshold_fails(tmp_path: Path) -> None:
    """A configured threshold outside zero to one should be rejected."""

    path = tmp_path / "thresholds.yaml"
    _write(
        path,
        {
            "balanced": {"PERSON": 1.2},
            "strict": {},
        },
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_threshold_profile("balanced", path=path)
