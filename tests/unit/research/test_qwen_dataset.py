"""Tests for Qwen training dataset preparation."""

from collections.abc import Sequence

from research.data.models import DatasetExample
from research.train.qwen_data import QwenMessage
from research.train.qwen_dataset import prepare_qwen_dataset
from research.train.qwen_tokens import IGNORE_INDEX


class FakeTokenizer:
    """Character-level chat tokenizer for dataset preparation tests."""

    def apply_chat_template(
        self,
        conversation: Sequence[QwenMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False

        text = ""

        for message in conversation:
            text += f"<{message['role']}>{message['content']}</{message['role']}>"

        if add_generation_prompt:
            text += "<assistant>"

        return text

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ):
        assert add_special_tokens is False
        assert return_offsets_mapping is True

        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _example(
    example_id: str,
    text: str,
) -> DatasetExample:
    return DatasetExample(
        example_id=example_id,
        text=text,
        spans=(),
        source="test",
        language="en",
    )


def test_prepares_training_records() -> None:
    """Examples fitting the context should become trainer records."""

    records, overflow_count = prepare_qwen_dataset(
        [_example("one", "hello")],
        FakeTokenizer(),
        max_length=2000,
    )

    assert len(records) == 1
    assert overflow_count == 0

    record = records[0]

    assert len(record["input_ids"]) == len(record["labels"])
    assert len(record["input_ids"]) == len(record["attention_mask"])
    assert any(label != IGNORE_INDEX for label in record["labels"])


def test_skips_overlong_examples_and_counts_them() -> None:
    """Context overflows should be explicit rather than truncated."""

    records, overflow_count = prepare_qwen_dataset(
        [
            _example("short", "hello"),
            _example("long", "x" * 3000),
        ],
        FakeTokenizer(),
        max_length=2000,
    )

    assert len(records) == 1
    assert overflow_count == 1


def test_preserves_input_order() -> None:
    """Dataset preparation should not reshuffle examples."""

    records, overflow_count = prepare_qwen_dataset(
        [
            _example("one", "a"),
            _example("two", "bb"),
        ],
        FakeTokenizer(),
        max_length=2000,
    )

    assert overflow_count == 0
    assert len(records) == 2
    assert len(records[0]["input_ids"]) < len(records[1]["input_ids"])


def test_rejects_non_positive_max_length() -> None:
    """Dataset preparation requires a positive context length."""

    try:
        prepare_qwen_dataset(
            [],
            FakeTokenizer(),
            max_length=0,
        )
    except ValueError as error:
        assert str(error) == "max_length must be positive"
    else:
        raise AssertionError("expected ValueError")
