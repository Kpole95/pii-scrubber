"""Inspect real Qwen chat-token lengths for normalized training data."""

from argparse import ArgumentParser
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from research.data.artifacts import load_dataset_split
from research.data.models import DatasetExample
from research.train.qwen_data import build_qwen_messages


def main() -> None:
    """Print token-length statistics for Qwen training examples."""

    args = _parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )
    artifact = load_dataset_split(args.input)

    print(f"Model: {args.model_name}")
    print(f"Tokenizer fast: {tokenizer.is_fast}")
    print()

    _report_split(
        "train",
        artifact.split.train,
        tokenizer,
    )
    _report_split(
        "validation",
        artifact.split.validation,
        tokenizer,
    )


def _report_split(
    name: str,
    examples: tuple[DatasetExample, ...],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Measure one frozen dataset split using the real chat template."""

    total_lengths: list[int] = []
    target_lengths: list[int] = []

    thresholds = {
        512: 0,
        1024: 0,
        2048: 0,
        4096: 0,
    }

    longest: tuple[int, DatasetExample] | None = None

    for example in examples:
        messages = build_qwen_messages(example)
        conversation: list[dict[str, str]] = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
        ]

        prompt_text = tokenizer.apply_chat_template(
            conversation[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not isinstance(prompt_text, str) or not isinstance(full_text, str):
            raise TypeError("chat template must render text when tokenize=False")

        if not full_text.startswith(prompt_text):
            raise ValueError(
                f"chat-template text prefix mismatch for example {example.example_id!r}"
            )

        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if not isinstance(input_ids, list) or not isinstance(offsets, list):
            raise TypeError("tokenizer must return list inputs for one example")

        total_length = len(input_ids)
        assistant_start = len(prompt_text)
        target_length = sum(1 for _, end in offsets if end > assistant_start)

        total_lengths.append(total_length)
        target_lengths.append(target_length)

        for threshold in thresholds:
            if total_length > threshold:
                thresholds[threshold] += 1

        if longest is None or total_length > longest[0]:
            longest = (total_length, example)

    print(f"{name}: {len(examples):,} examples")
    print("  total tokens")
    _print_stats(total_lengths)

    print("  assistant target tokens")
    _print_stats(target_lengths)

    for threshold, count in thresholds.items():
        percentage = 100 * count / len(examples) if examples else 0.0
        print(f"  > {threshold:4d}: {count:6,d} ({percentage:6.2f}%)")

    if longest is not None:
        length, example = longest
        print(f"  longest example: {example.example_id}")
        print(f"  longest tokens:  {length}")
        print(f"  text chars:      {len(example.text)}")

    print()


def _print_stats(values: list[int]) -> None:
    """Print deterministic token-length summary statistics."""

    if not values:
        print("    no examples")
        return

    ordered = sorted(values)

    print(f"    min: {ordered[0]}")
    print(f"    p50: {_percentile(ordered, 0.50)}")
    print(f"    p95: {_percentile(ordered, 0.95)}")
    print(f"    p99: {_percentile(ordered, 0.99)}")
    print(f"    max: {ordered[-1]}")
    print(f"    median: {median(ordered):.1f}")


def _percentile(
    ordered: list[int],
    fraction: float,
) -> int:
    """Return a nearest-rank percentile from sorted integer values."""

    if not ordered:
        raise ValueError("ordered values must not be empty")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")

    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _parser() -> ArgumentParser:
    """Build the Qwen dataset-inspection CLI."""

    parser = ArgumentParser(
        description="Inspect Qwen chat-token lengths.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/data_artifacts/ai4privacy"),
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    return parser


if __name__ == "__main__":
    main()
