"""Prepare normalized examples for Qwen LoRA training."""

from collections.abc import Sequence
from typing import TypedDict

from research.data.models import DatasetExample
from research.train.qwen_data import build_qwen_messages
from research.train.qwen_tokens import ChatTokenizer, tokenize_qwen_messages


class QwenTrainingRecord(TypedDict):
    """One tokenized causal-LM training example."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def prepare_qwen_dataset(
    examples: Sequence[DatasetExample],
    tokenizer: ChatTokenizer,
    *,
    max_length: int,
) -> tuple[list[QwenTrainingRecord], int]:
    """Tokenize examples and explicitly skip those exceeding context length."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    records: list[QwenTrainingRecord] = []
    overflow_count = 0

    for example in examples:
        messages = build_qwen_messages(example)

        try:
            tokenized = tokenize_qwen_messages(
                messages,
                tokenizer,
                max_length=max_length,
            )
        except ValueError as error:
            if "exceeding max_length" not in str(error):
                raise

            overflow_count += 1
            continue

        records.append(
            QwenTrainingRecord(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                labels=tokenized["labels"],
            )
        )

    return records, overflow_count
