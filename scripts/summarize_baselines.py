"""Write a Markdown summary for baseline evaluation reports."""

from argparse import ArgumentParser
from pathlib import Path

from research.eval.report_io import read_report


def main() -> None:
    """Read reports and write one Markdown comparison table."""
    args = _parser().parse_args()

    rows = [_row(path) for path in args.inputs]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        _markdown(rows),
        encoding="utf-8",
    )


def _row(
    path: Path,
) -> tuple[str, dict[str, object]]:
    """Return a report name and its decoded JSON."""
    return path.stem, read_report(path)


def _markdown(
    rows: list[tuple[str, dict[str, object]]],
) -> str:
    """Build a compact Markdown comparison table."""
    lines = [
        "# Baseline Evaluation",
        "",
        "| Run | Examples | Leak rate | Exact F1 | Partial F1 | Over-redaction |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, report in rows:
        exact = report["exact"]
        partial = report["partial"]

        if not isinstance(exact, dict):
            raise TypeError("exact report must be a mapping")

        if not isinstance(partial, dict):
            raise TypeError("partial report must be a mapping")

        lines.append(
            "| "
            f"{name} | "
            f"{report['examples']} | "
            f"{_number(report['leak_rate'], 'leak_rate'):.4f} | "
            f"{_number(exact['f1'], 'exact f1'):.4f} | "
            f"{_number(partial['f1'], 'partial f1'):.4f} | "
            f"{_number(report['over_redaction_rate'], 'over_redaction_rate'):.4f} |"
        )

    return "\n".join(lines) + "\n"


def _number(
    value: object,
    name: str,
) -> float:
    """Return a numeric report value as float."""
    if not isinstance(
        value,
        int | float,
    ):
        raise TypeError(f"{name} must be numeric")

    return float(value)


def _parser() -> ArgumentParser:
    """Build the summary CLI parser."""
    parser = ArgumentParser(
        description="Summarize baseline evaluation reports.",
    )

    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser


if __name__ == "__main__":
    main()
