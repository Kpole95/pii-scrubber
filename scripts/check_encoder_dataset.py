"""Inspect encoder preparation on the real Ai4Privacy artifact."""

from transformers import AutoTokenizer

from research.data.artifacts import load_dataset_split
from research.train.encoder_data import (
    build_label_mapping,
    prepare_encoder_example,
    window_example,
)


def main() -> None:
    """Inspect real training examples with the encoder tokenizer."""
    artifact = load_dataset_split("research/data_artifacts/ai4privacy")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    labels = build_label_mapping(artifact.split.train)

    lengths = []
    overlong = 0

    for example in artifact.split.train:
        try:
            aligned, _ = prepare_encoder_example(
                example,
                tokenizer,
                labels,
                strategy="all_subwords",
                max_length=512,
            )
            lengths.append(aligned.token_count)
        except ValueError as error:
            if "exceeds max_length" not in str(error):
                raise
            windows = window_example(example, tokenizer, max_length=512, overlap=64)
            print("WINDOWED", example.example_id, len(windows))

    print("train_examples:", len(artifact.split.train))
    print("labels:", len(labels))
    print("max_tokens:", max(lengths))
    print("overlong:", overlong)
    print("overlong_rate:", overlong / len(artifact.split.train))


if __name__ == "__main__":
    main()
