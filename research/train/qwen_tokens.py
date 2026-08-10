"""Tokenize Qwen chat examples with assistant-only training labels."""

from collections.abc import Sequence
from typing import Protocol, TypedDict

from research.train.qwen_data import QwenMessage

IGNORE_INDEX = -100


class TokenizedText(TypedDict):
    """Minimal tokenizer output needed for assistant masking."""

    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]


class ChatTokenizer(Protocol):
    """Minimal tokenizer contract needed for Qwen training preparation."""

    def apply_chat_template(
        self,
        conversation: Sequence[QwenMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> TokenizedText: ...


def tokenize_qwen_messages(
    messages: Sequence[QwenMessage],
    tokenizer: ChatTokenizer,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize one chat example with loss only on assistant output."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    _validate_messages(messages)

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if not full_text.startswith(prompt_text):
        raise ValueError("chat-template prompt text is not a prefix of the full example")

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offsets = list(encoded["offset_mapping"])

    if len(input_ids) > max_length:
        raise ValueError(
            f"tokenized Qwen example has {len(input_ids)} tokens, exceeding max_length={max_length}"
        )

    assistant_start = len(prompt_text)

    labels = [
        token_id if end > assistant_start else IGNORE_INDEX
        for token_id, (_, end) in zip(input_ids, offsets, strict=True)
    ]

    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("chat template produced no assistant training tokens")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _validate_messages(messages: Sequence[QwenMessage]) -> None:
    """Require the system/user/assistant training conversation structure."""

    if len(messages) != 3:
        raise ValueError("Qwen training examples must contain exactly three messages")

    roles = tuple(message.get("role") for message in messages)
    if roles != ("system", "user", "assistant"):
        raise ValueError("Qwen messages must use system, user, assistant role order")

    if any(not isinstance(message.get("content"), str) for message in messages):
        raise TypeError("Qwen message content must be strings")
