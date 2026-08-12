"""Evaluate frozen encoder threshold profiles from one inference pass."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from datasets import load_dataset

from pii_scrub.threshold_profiles import load_threshold_profile
from research.data.artifacts import load_dataset_split
from research.data.conll import load_conll2003_record
from research.data.models import DatasetExample
from research.data.ood_io import load_ood_jsonl
from research.eval.encoder import load_encoder_predictor
from research.eval.encoder_profiles import (
    evaluate_cached,
    predict_examples,
    write_prediction_cache,
)
from research.eval.report_io import write_report


def main() -> None:
    """Run encoder once and evaluate raw, balanced, and strict outputs."""

    args = _parser().parse_args()
    examples = _load_examples(
        args.dataset,
        args.input,
        args.split,
        args.data_file,
    )
    entities = {"PERSON"} if args.dataset == "conll" else None
    predictor = load_encoder_predictor(args.model_path, device=args.device)
    predictions = predict_examples(
        examples,
        predictor,
        entities=entities,
        progress_every=args.progress_every,
    )

    if args.predictions_output is not None:
        write_prediction_cache(examples, predictions, args.predictions_output)
        print(f"wrote {args.predictions_output}", flush=True)

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "examples": len(examples),
        "profiles": {
            "raw": evaluate_cached(
                examples,
                predictions,
                entities=entities,
            ),
            "balanced": evaluate_cached(
                examples,
                predictions,
                thresholds=load_threshold_profile("balanced"),
                entities=entities,
            ),
            "strict": evaluate_cached(
                examples,
                predictions,
                thresholds=load_threshold_profile("strict"),
                entities=entities,
            ),
        },
    }
    write_report(report, args.output)
    print(f"wrote {args.output}", flush=True)
    _print_summary(report)


def _load_examples(
    dataset: str,
    path: Path | None,
    split: str | None,
    data_file: str | None,
) -> Sequence[DatasetExample]:
    """Load one supported frozen benchmark dataset."""

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


def _print_summary(report: dict[str, object]) -> None:
    """Print compact raw and threshold-profile results."""

    profiles = report["profiles"]
    if not isinstance(profiles, dict):
        raise TypeError("profiles must be a dictionary")

    print()
    print(f"{'PROFILE':<12}{'EXACT F1':>12}{'PARTIAL F1':>12}{'LEAK':>12}{'OVER-RED':>12}")
    print("=" * 60)

    for name in ("raw", "balanced", "strict"):
        result = profiles[name]
        if not isinstance(result, dict):
            raise TypeError("profile result must be a dictionary")

        exact = result["exact"]
        partial = result["partial"]
        if not isinstance(exact, dict):
            raise TypeError("exact metrics must be a dictionary")
        if not isinstance(partial, dict):
            raise TypeError("partial metrics must be a dictionary")

        print(
            f"{name:<12}"
            f"{float(exact['f1']):12.6f}"
            f"{float(partial['f1']):12.6f}"
            f"{float(result['leak_rate']):12.6f}"
            f"{float(result['over_redaction_rate']):12.6f}"
        )


def _parser() -> ArgumentParser:
    """Build the frozen-profile evaluation command-line parser."""

    parser = ArgumentParser(
        description=(
            "Evaluate raw, balanced, and strict encoder predictions from one inference pass."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument(
        "--dataset",
        choices=("artifact", "ood", "conll"),
        required=True,
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


if __name__ == "__main__":
    main()
