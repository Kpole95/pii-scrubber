"""Tests for evaluation report persistence."""

import json
from pathlib import Path

import pytest

from research.eval.report_io import (
    read_report,
    write_report,
)


def test_write_report_round_trip(
    tmp_path: Path,
) -> None:
    """A written report should load unchanged.

    Example:
        ``{"examples": 2}`` survives a JSON round trip.
    """
    path = tmp_path / "nested" / "report.json"
    report = {
        "examples": 2,
        "leak_rate": 0.25,
    }

    write_report(
        report,
        path,
    )

    assert read_report(path) == report


def test_write_report_is_valid_json(
    tmp_path: Path,
) -> None:
    """Written output should be standard JSON.

    Example:
        Python's JSON loader can read the saved file.
    """
    path = tmp_path / "report.json"

    write_report(
        {"examples": 1},
        path,
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    assert data["examples"] == 1


def test_read_report_rejects_non_object(
    tmp_path: Path,
) -> None:
    """Top-level JSON must be an object.

    Example:
        A JSON list raises ``ValueError``.
    """
    path = tmp_path / "report.json"
    path.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        read_report(path)
