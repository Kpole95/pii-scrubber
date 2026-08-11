"""Calibrate encoder confidence scores on a normalized validation split."""

import json
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.data.artifacts import load_dataset_split
from research.data.models import DatasetExample
from research.eval.calibration import (
    calibration_error,
    per_entity_calibration_error,
    sweep_thresholds,
    threshold_grid,
)
from research.eval.encoder import load_encoder_predictor
from research.eval.report_io import write_report


def main() -> None:
    """Run encoder inference once and write validation calibration metrics."""

    args = _parser().parse_args()

    artifact = load_dataset_split(args.input)
    examples = _select_split(
        artifact.split.train,
        artifact.split.validation,
        artifact.split.test,
        args.split,
    )

    if args.limit is not None:
        examples = examples[: args.limit]

    predictor = load_encoder_predictor(
        args.model_path,
        device=args.device,
    )

    predictions = []
    total = len(examples)

    for index, example in enumerate(examples, start=1):
        predictions.append(
            predictor(
                example.text,
                None,
            )
        )

        if index == total or index % args.progress_every == 0:
            print(f"predicted {index:,}/{total:,}", flush=True)

    if args.predictions_output is not None:
        _write_prediction_cache(
            examples,
            predictions,
            args.predictions_output,
        )
        print(f"wrote {args.predictions_output}", flush=True)

    thresholds = threshold_grid(step=args.step)

    sweep = sweep_thresholds(
        examples,
        predictions,
        thresholds,
    )

    report: dict[str, object] = {
        "dataset": "ai4privacy",
        "split": args.split,
        "examples": len(examples),
        "bins": args.bins,
        "threshold_step": args.step,
        "ece": calibration_error(
            examples,
            predictions,
            bins=args.bins,
        ),
        "per_entity_ece": per_entity_calibration_error(
            examples,
            predictions,
            bins=args.bins,
        ),
        "sweep": [
            {
                "threshold": score.threshold,
                "leak_rate": score.leak_rate,
                "exact_precision": score.precision,
                "exact_recall": score.recall,
                "exact_f1": score.f1,
                "over_redaction_rate": score.over_redaction_rate,
                "predictions": score.predictions,
            }
            for score in sweep
        ],
    }

    write_report(report, args.output)
    print(f"wrote {args.output}", flush=True)


def _write_prediction_cache(
    examples: tuple[DatasetExample, ...],
    predictions: list[Any],
    output: Path,
) -> None:
    """Write normalized examples and scored predictions as JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for index, (example, example_predictions) in enumerate(
            zip(examples, predictions, strict=True)
        ):
            record = {
                "example_index": index,
                "example": asdict(example),
                "predictions": [
                    asdict(prediction)
                    for prediction in example_predictions
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


def _select_split(
    train: tuple[DatasetExample, ...],
    validation: tuple[DatasetExample, ...],
    test: tuple[DatasetExample, ...],
    split: str,
) -> tuple[DatasetExample, ...]:
    """Return the requested normalized artifact split."""

    if split == "train":
        return train

    if split == "validation":
        return validation

    return test


def _parser() -> ArgumentParser:
    """Build the Stage 12 encoder calibration CLI."""

    parser = ArgumentParser(
        description=(
            "Calibrate encoder confidence scores on "
            "Ai4Privacy validation data."
        ),
    )

    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument(
        "--predictions-output",
        type=Path,
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )

    parser.add_argument("--device", default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--limit", type=int)

    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
    )

    return parser


if __name__ == "__main__":
    main()
