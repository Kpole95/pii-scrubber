"""Evaluate frozen encoder threshold profiles from one inference pass."""

import json
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

from pii_scrub.calibration import apply_thresholds
from pii_scrub.detectors.encoder import EncoderDetector
from pii_scrub.threshold_profiles import load_threshold_profile
from pii_scrub.types import DetectedSpan
from research.data.artifacts import load_dataset_split
from research.data.conll import load_conll2003_record
from research.data.models import DatasetExample
from research.data.ood_io import load_ood_jsonl
from research.eval.encoder import load_encoder_predictor
from research.eval.report_io import write_report
from research.eval.run_baseline import evaluate_baseline

Predictions = tuple[tuple[DetectedSpan, ...], ...]


def main() -> None:
    """Run encoder once and evaluate raw, balanced, and strict outputs."""

    args = _parser().parse_args()

    examples = _load_examples(
        args.dataset,
        args.input,
        args.split,
        args.data_file,
    )

    entities = (
        {"PERSON"}
        if args.dataset == "conll"
        else None
    )

    predictor = load_encoder_predictor(
        args.model_path,
        device=args.device,
    )

    predictions = _predict(
        examples,
        predictor,
        entities=entities,
        progress_every=args.progress_every,
    )

    if args.predictions_output is not None:
        _write_prediction_cache(
            examples,
            predictions,
            args.predictions_output,
        )

        print(
            f"wrote {args.predictions_output}",
            flush=True,
        )

    balanced = load_threshold_profile(
        "balanced",
    )
    strict = load_threshold_profile(
        "strict",
    )

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "examples": len(examples),
        "profiles": {
            "raw": _evaluate_cached(
                examples,
                predictions,
                entities=entities,
            ),
            "balanced": _evaluate_cached(
                examples,
                predictions,
                thresholds=balanced,
                entities=entities,
            ),
            "strict": _evaluate_cached(
                examples,
                predictions,
                thresholds=strict,
                entities=entities,
            ),
        },
    }

    write_report(
        report,
        args.output,
    )

    print(
        f"wrote {args.output}",
        flush=True,
    )

    _print_summary(report)


def _predict(
    examples: Sequence[DatasetExample],
    predictor: Callable[
        [str, set[str] | None],
        list[DetectedSpan],
    ],
    *,
    entities: set[str] | None,
    progress_every: int,
) -> Predictions:
    """Run encoder inference exactly once per example."""

    predictions: list[tuple[DetectedSpan, ...]] = []
    total = len(examples)

    for index, example in enumerate(
        examples,
        start=1,
    ):
        predictions.append(
            tuple(
                predictor(
                    example.text,
                    entities,
                )
            )
        )

        if (
            index == total
            or index % progress_every == 0
        ):
            print(
                f"predicted {index:,}/{total:,}",
                flush=True,
            )

    return tuple(predictions)


def _evaluate_cached(
    examples: Sequence[DatasetExample],
    predictions: Predictions,
    *,
    thresholds: dict[str, float] | None = None,
    entities: set[str] | None,
) -> dict[str, object]:
    """Evaluate cached predictions without rerunning the encoder."""

    index = 0

    def replay(
        text: str,
        requested_entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        nonlocal index

        if index >= len(predictions):
            raise RuntimeError(
                "prediction cache exhausted"
            )

        example = examples[index]

        if text != example.text:
            raise ValueError(
                "prediction cache and examples are out of order"
            )

        output = list(
            predictions[index]
        )
        index += 1

        if thresholds is not None:
            output = apply_thresholds(
                output,
                thresholds,
            )

        if requested_entities is not None:
            output = [
                prediction
                for prediction in output
                if prediction.entity_type
                in requested_entities
            ]

        return output

    detector = EncoderDetector(
        replay,
    )

    report = evaluate_baseline(
        detector,
        examples,
        entities=entities,
    )

    if index != len(predictions):
        raise RuntimeError(
            "prediction cache was not fully consumed"
        )

    return report


def _write_prediction_cache(
    examples: Sequence[DatasetExample],
    predictions: Predictions,
    output: Path,
) -> None:
    """Write examples and scored predictions as JSONL."""

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for index, (
            example,
            example_predictions,
        ) in enumerate(
            zip(
                examples,
                predictions,
                strict=True,
            )
        ):
            record = {
                "example_index": index,
                "example": asdict(example),
                "predictions": [
                    asdict(prediction)
                    for prediction
                    in example_predictions
                ],
            }

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")


def _load_examples(
    dataset: str,
    path: Path | None,
    split: str | None,
    data_file: str | None,
) -> Sequence[DatasetExample]:
    """Load one supported Stage 12 benchmark."""

    if dataset == "ood":
        if path is None:
            raise ValueError(
                "--input is required for OOD"
            )

        return load_ood_jsonl(
            path,
        )

    if dataset == "artifact":
        if path is None:
            raise ValueError(
                "--input is required for artifact datasets"
            )

        if split is None:
            raise ValueError(
                "--split is required for artifact datasets"
            )

        artifact = load_dataset_split(
            path,
        )

        if split == "train":
            return artifact.split.train

        if split == "validation":
            return artifact.split.validation

        return artifact.split.test

    if data_file is None:
        raise ValueError(
            "--data-file is required for CoNLL"
        )

    rows = load_dataset(
        "parquet",
        data_files=data_file,
        split=split or "train",
    )

    return tuple(
        load_conll2003_record(row)
        for row in rows
    )


def _print_summary(
    report: dict[str, object],
) -> None:
    """Print compact profile results."""

    profiles = report["profiles"]

    if not isinstance(profiles, dict):
        raise TypeError(
            "profiles must be a dictionary"
        )

    print()
    print(
        f"{'PROFILE':<12}"
        f"{'EXACT F1':>12}"
        f"{'PARTIAL F1':>12}"
        f"{'LEAK':>12}"
        f"{'OVER-RED':>12}"
    )
    print("=" * 60)

    for name in (
        "raw",
        "balanced",
        "strict",
    ):
        result = profiles[name]

        if not isinstance(result, dict):
            raise TypeError(
                "profile result must be a dictionary"
            )

        exact = result["exact"]
        partial = result["partial"]

        if not isinstance(exact, dict):
            raise TypeError(
                "exact metrics must be a dictionary"
            )

        if not isinstance(partial, dict):
            raise TypeError(
                "partial metrics must be a dictionary"
            )

        print(
            f"{name:<12}"
            f"{float(exact['f1']):12.6f}"
            f"{float(partial['f1']):12.6f}"
            f"{float(result['leak_rate']):12.6f}"
            f"{float(result['over_redaction_rate']):12.6f}"
        )


def _parser() -> ArgumentParser:
    """Build the Stage 12 frozen-profile evaluation CLI."""

    parser = ArgumentParser(
        description=(
            "Evaluate raw, balanced, and strict encoder "
            "predictions from one inference pass."
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--predictions-output",
        type=Path,
    )

    parser.add_argument(
        "--dataset",
        choices=(
            "artifact",
            "ood",
            "conll",
        ),
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "test",
        ),
    )

    parser.add_argument(
        "--data-file",
        type=str,
    )

    parser.add_argument(
        "--device",
        type=str,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    return parser


if __name__ == "__main__":
    main()