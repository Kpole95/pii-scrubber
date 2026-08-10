"""Inspect exact matches broken by the PERSON merge ablation."""

from argparse import ArgumentParser
from pathlib import Path

from research.data.artifacts import load_dataset_split
from research.eval.ablation import merge_separated_same_type
from research.eval.encoder import load_encoder_predictor


def main() -> None:
    """Print Ai4Privacy PERSON cases where merging breaks raw exact matches."""

    args = _parser().parse_args()
    artifact = load_dataset_split(args.data)
    predict = load_encoder_predictor(args.model_path, device=args.device)

    regressions = 0

    for example in artifact.split.test:
        gold = tuple(span for span in example.spans if span.entity_type == "PERSON")
        if not gold:
            continue

        raw = predict(example.text, {"PERSON"})
        merged = merge_separated_same_type(
            example.text,
            raw,
            entity_types={"PERSON"},
        )

        raw_exact = {(span.start, span.end, span.entity_type) for span in raw}
        merged_exact = {(span.start, span.end, span.entity_type) for span in merged}

        lost = [
            span
            for span in gold
            if (span.start, span.end, span.entity_type) in raw_exact
            and (span.start, span.end, span.entity_type) not in merged_exact
        ]

        if not lost:
            continue

        regressions += len(lost)

        if regressions > args.limit:
            break

        print()
        print("TEXT:", example.text)
        print(
            "GOLD:",
            [
                (
                    example.text[span.start : span.end],
                    span.start,
                    span.end,
                )
                for span in gold
            ],
        )
        print(
            "RAW:",
            [
                (
                    example.text[span.start : span.end],
                    span.start,
                    span.end,
                    span.score,
                )
                for span in raw
            ],
        )
        print(
            "MERGED:",
            [
                (
                    example.text[span.start : span.end],
                    span.start,
                    span.end,
                    span.score,
                )
                for span in merged
            ],
        )

    print()
    print("Displayed regressions:", min(regressions, args.limit))


def _parser() -> ArgumentParser:
    """Build the regression-inspection CLI."""

    parser = ArgumentParser(
        description="Inspect PERSON merge regressions on Ai4Privacy.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("research/data_artifacts/ai4privacy"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
    )
    return parser


if __name__ == "__main__":
    main()
