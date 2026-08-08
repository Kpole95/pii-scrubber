"""Run one baseline evaluation from CLI arguments."""

from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from pathlib import Path

from datasets import load_dataset

from pii_scrub.detectors.base import Detector
from pii_scrub.detectors.presidio import PresidioDetector
from pii_scrub.detectors.regex import RegexDetector
from research.data.artifacts import load_dataset_split
from research.data.conll import load_conll2003_record
from research.data.models import DatasetExample
from research.data.ood_io import load_ood_jsonl
from research.eval.report_io import write_report
from research.eval.run_baseline import evaluate_baseline

DETECTORS: dict[str, Callable[[], Detector]] = {
    "regex": RegexDetector,
    "presidio": PresidioDetector,
}


def main() -> None:
    """Parse arguments, evaluate one detector, and write JSON.

    Example:
        ``python -m scripts.run_evaluation --detector regex ...``
    """
    args = _parser().parse_args()

    detector = DETECTORS[args.detector]()
    examples = _load_examples(
        args.dataset,
        args.input,
        args.split,
        args.data_file,
    )

    report = evaluate_baseline(
        detector,
        examples,
    )

    write_report(
        report,
        args.output,
    )


def _load_examples(
    dataset: str,
    path: Path | None,
    split: str | None,
    data_file: str | None,
) -> Sequence[DatasetExample]:
    """Load normalized examples for one supported source."""
    if dataset == "ood":
        if path is None:
            raise ValueError("--input is required for OOD")

        return load_ood_jsonl(path)

    if dataset == "artifact":
        if path is None:
            raise ValueError("--input is required for artifact datasets")

        if split is None:
            raise ValueError("--split is required for artifact datasets")

        artifact = load_dataset_split(path)

        if split == "train":
            return artifact.split.train

        if split == "validation":
            return artifact.split.validation

        return artifact.split.test

    if data_file is None:
        raise ValueError("--data-file is required for CoNLL")

    rows = load_dataset(
        "parquet",
        data_files=data_file,
        split=split or "train",
    )

    return tuple(load_conll2003_record(row) for row in rows)


def _parser() -> ArgumentParser:
    """Build the baseline evaluation CLI parser."""
    parser = ArgumentParser(
        description="Run one PII baseline evaluation.",
    )

    parser.add_argument(
        "--detector",
        choices=tuple(DETECTORS),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=("ood", "artifact", "conll"),
        default="ood",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
    )
    parser.add_argument(
        "--data-file",
        type=str,
    )
    parser.add_argument(
        "--input",
        type=Path,
    )

    return parser


if __name__ == "__main__":
    main()
