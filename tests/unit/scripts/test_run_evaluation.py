"""Tests for the baseline evaluation CLI."""

from pathlib import Path

from scripts.run_evaluation import _parser


def test_parser_accepts_required_arguments(
    tmp_path: Path,
) -> None:
    """The CLI should parse one complete evaluation run."""
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "report.json"

    args = _parser().parse_args(
        [
            "--detector",
            "regex",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert args.detector == "regex"
    assert args.input == input_path
    assert args.output == output_path


def test_parser_accepts_artifact_split(
    tmp_path: Path,
) -> None:
    """The CLI should accept an artifact dataset split."""
    args = _parser().parse_args(
        [
            "--detector",
            "regex",
            "--dataset",
            "artifact",
            "--input",
            str(tmp_path / "artifact"),
            "--split",
            "test",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.dataset == "artifact"
    assert args.split == "test"


def test_parser_accepts_conll_source(
    tmp_path: Path,
) -> None:
    """The CLI should accept a CoNLL parquet source."""
    args = _parser().parse_args(
        [
            "--detector",
            "regex",
            "--dataset",
            "conll",
            "--data-file",
            "example.parquet",
            "--split",
            "train",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.dataset == "conll"
    assert args.data_file == "example.parquet"
