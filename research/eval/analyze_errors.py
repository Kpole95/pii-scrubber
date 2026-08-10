"""Analyze encoder span errors on normalized evaluation examples."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from research.data.conll import load_conll2003_record
from research.eval.encoder import load_encoder_predictor
from research.eval.errors import classify_gold_errors, false_positive_spans


def main() -> None:
    """Run PERSON-only CoNLL error analysis."""

    args = _parser().parse_args()
    rows = load_dataset(
        "parquet",
        data_files=str(args.data_file),
        split="train",
    )
    predict = load_encoder_predictor(
        args.model_path,
        device=args.device,
    )

    counts: Counter[str] = Counter()
    false_positives = 0
    examples: dict[str, list[str]] = {}

    for row in rows:
        example = load_conll2003_record(row)
        predictions = predict(example.text, {"PERSON"})
        errors = classify_gold_errors(example.text, example.spans, predictions)

        false_positives += len(
            false_positive_spans(
                example.spans,
                predictions,
            )
        )

        for error in errors:
            counts[error.kind] += 1

            if len(examples.setdefault(error.kind, [])) >= args.examples:
                continue

            gold_text = example.text[error.gold.start : error.gold.end]
            predicted = [
                {
                    "text": example.text[prediction.start : prediction.end],
                    "start": prediction.start,
                    "end": prediction.end,
                    "score": prediction.score,
                }
                for prediction in error.predictions
            ]

            examples[error.kind].append(
                f"TEXT: {example.text}\n"
                f"GOLD: {gold_text!r} [{error.gold.start}:{error.gold.end}]\n"
                f"PRED: {predicted}"
            )

    print("Gold error counts")
    print("=================")

    for kind, count in counts.most_common():
        print(f"{kind:15} {count}")

    print(f"{'false_positive':15} {false_positives}")

    for kind in sorted(examples):
        print()
        print(kind.upper())
        print("=" * len(kind))

        for item in examples[kind]:
            print()
            print(item)


def _parser() -> ArgumentParser:
    """Build the error-analysis CLI."""

    parser = ArgumentParser(description="Analyze encoder CoNLL PERSON errors.")
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
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
        "--examples",
        type=int,
        default=3,
    )

    return parser


if __name__ == "__main__":
    main()
