"""Read and write evaluation reports."""

import json
from pathlib import Path
from typing import Any


def write_report(
    report: dict[str, object],
    path: Path,
) -> None:
    """Write one evaluation report as formatted JSON.

    Example:
        ``write_report(report, Path("result.json"))`` saves the report.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_report(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON evaluation report.

    Example:
        ``read_report(Path("result.json"))`` returns the saved data.
    """
    data = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(data, dict):
        raise ValueError("evaluation report must be a JSON object")

    return data
