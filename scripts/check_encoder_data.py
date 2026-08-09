"""Smoke-check encoder preparation with a real tokenizer."""

from transformers import AutoTokenizer

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.train.encoder_data import build_label_mapping, prepare_encoder_example


def main() -> None:
    """Prepare one example with the configured encoder tokenizer."""
    example = DatasetExample(
        "smoke",
        "Email ana@example.com today.",
        (CharacterSpan(6, 21, "EMAIL"),),
        "smoke",
        "en",
    )
    labels = build_label_mapping((example,))
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)

    aligned, label_ids = prepare_encoder_example(
        example,
        tokenizer,
        labels,
        strategy="all_subwords",
        max_length=512,
    )

    print("tokens:", aligned.token_count)
    print("labels:", aligned.token_labels)
    print("label_ids:", label_ids)
    print("mapping:", labels)


if __name__ == "__main__":
    main()
