"""Select frozen encoder threshold profiles from validation predictions."""

import json
from argparse import ArgumentParser
from pathlib import Path

from research.eval.calibration import threshold_grid
from research.eval.encoder_profiles import load_prediction_cache
from research.eval.report_io import write_report
from research.eval.threshold_profiles import (
    ThresholdProfileScore,
    optimize_threshold_profile,
    score_threshold_profile,
)

STRICT_ANCHOR = 0.35
BALANCED_ANCHOR = 0.60


def main() -> None:
    """Select strict and balanced profiles from validation predictions."""

    args = _parser().parse_args()
    examples, predictions = load_prediction_cache(args.predictions)
    entities = sorted(
        {span.entity_type for example in examples for span in example.spans}
        | {prediction.entity_type for output in predictions for prediction in output}
    )
    thresholds = threshold_grid(step=args.step)

    strict_anchor = {entity: STRICT_ANCHOR for entity in entities}
    balanced_anchor = {entity: BALANCED_ANCHOR for entity in entities}
    strict_anchor_score = score_threshold_profile(
        examples,
        predictions,
        strict_anchor,
    )
    balanced_anchor_score = score_threshold_profile(
        examples,
        predictions,
        balanced_anchor,
    )

    print("optimizing strict profile...", flush=True)
    strict_profile, strict_score = optimize_threshold_profile(
        examples,
        predictions,
        entities,
        thresholds,
        initial_threshold=STRICT_ANCHOR,
        mode="strict",
        passes=args.passes,
    )

    print("optimizing balanced profile...", flush=True)
    balanced_profile, balanced_score = optimize_threshold_profile(
        examples,
        predictions,
        entities,
        thresholds,
        initial_threshold=BALANCED_ANCHOR,
        mode="balanced",
        passes=args.passes,
    )

    profiles = {
        "balanced": balanced_profile,
        "strict": strict_profile,
    }
    _write_threshold_config(args.thresholds_output, profiles)
    report = _selection_report(
        len(examples),
        args.step,
        args.passes,
        strict_anchor_score,
        balanced_anchor_score,
        strict_profile,
        strict_score,
        balanced_profile,
        balanced_score,
    )
    write_report(report, args.output)

    _print_profile("STRICT", strict_profile, strict_score)
    _print_profile("BALANCED", balanced_profile, balanced_score)
    print(f"wrote {args.thresholds_output}", flush=True)
    print(f"wrote {args.output}", flush=True)


def _selection_report(
    examples: int,
    step: float,
    passes: int,
    strict_anchor_score: ThresholdProfileScore,
    balanced_anchor_score: ThresholdProfileScore,
    strict_profile: dict[str, float],
    strict_score: ThresholdProfileScore,
    balanced_profile: dict[str, float],
    balanced_score: ThresholdProfileScore,
) -> dict[str, object]:
    """Build the reproducible threshold-selection report."""

    return {
        "dataset": "ai4privacy",
        "split": "validation",
        "examples": examples,
        "threshold_step": step,
        "passes": passes,
        "selection_policy": {
            "strict": (
                "coordinate search prioritizing minimum leak rate, then recall, "
                "then over-redaction and precision"
            ),
            "balanced": (
                "coordinate search prioritizing maximum exact-span F1, then lower "
                "leak rate and higher precision"
            ),
        },
        "global_anchors": {
            "strict": STRICT_ANCHOR,
            "balanced": BALANCED_ANCHOR,
        },
        "anchor_scores": {
            "strict": _score_dict(strict_anchor_score),
            "balanced": _score_dict(balanced_anchor_score),
        },
        "profiles": {
            "strict": {
                "thresholds": strict_profile,
                "metrics": _score_dict(strict_score),
            },
            "balanced": {
                "thresholds": balanced_profile,
                "metrics": _score_dict(balanced_score),
            },
        },
    }


def _score_dict(score: ThresholdProfileScore) -> dict[str, float | int]:
    """Serialize one aggregate profile score."""

    return {
        "leak_rate": score.leak_rate,
        "exact_precision": score.precision,
        "exact_recall": score.recall,
        "exact_f1": score.f1,
        "over_redaction_rate": score.over_redaction_rate,
        "predictions": score.predictions,
    }


def _write_threshold_config(
    path: Path,
    profiles: dict[str, dict[str, float]],
) -> None:
    """Write profiles as JSON-compatible YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profiles, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_profile(
    name: str,
    thresholds: dict[str, float],
    score: ThresholdProfileScore,
) -> None:
    """Print one selected profile and its validation metrics."""

    print()
    print(name)
    print("=" * 72)
    for entity, threshold in sorted(thresholds.items()):
        print(f"{entity:30} {threshold:.2f}")

    print()
    print(f"leak_rate:           {score.leak_rate:.6f}")
    print(f"exact_precision:     {score.precision:.6f}")
    print(f"exact_recall:        {score.recall:.6f}")
    print(f"exact_f1:            {score.f1:.6f}")
    print(f"over_redaction_rate: {score.over_redaction_rate:.6f}")
    print(f"predictions:         {score.predictions}")


def _parser() -> ArgumentParser:
    """Build the threshold-selection command-line parser."""

    parser = ArgumentParser(
        description=(
            "Select strict and balanced encoder thresholds from cached validation predictions."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        default=Path("configs/thresholds.yaml"),
    )
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--passes", type=int, default=3)
    return parser


if __name__ == "__main__":
    main()
