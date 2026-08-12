"""Calibrate encoder confidence scores on a normalized validation split."""

from argparse import ArgumentParser
from pathlib import Path

from research.data.artifacts import load_dataset_split
from research.data.models import DatasetExample
from research.eval.calibration import (
    calibration_error,
    per_entity_calibration_error,
    sweep_thresholds,
    threshold_grid,
)
from research.eval.encoder import load_encoder_predictor
from research.eval.encoder_profiles import predict_examples, write_prediction_cache
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

    predictor = load_encoder_predictor(args.model_path, device=args.device)
    predictions = predict_examples(
        examples,
        predictor,
        entities=None,
        progress_every=args.progress_every,
    )
    if args.predictions_output is not None:
        write_prediction_cache(examples, predictions, args.predictions_output)
        print(f"wrote {args.predictions_output}", flush=True)

    sweep = sweep_thresholds(
        examples,
        predictions,
        threshold_grid(step=args.step),
    )
    report = {
        "dataset": "ai4privacy",
        "split": args.split,
        "examples": len(examples),
        "bins": args.bins,
        "threshold_step": args.step,
        "ece": calibration_error(examples, predictions, bins=args.bins),
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
    """Build the encoder calibration command-line parser."""

    parser = ArgumentParser(
        description="Calibrate encoder confidence scores on Ai4Privacy validation data."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser


if __name__ == "__main__":
    main()
