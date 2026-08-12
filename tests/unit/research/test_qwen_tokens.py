"""Tests for Qwen chat tokenization and assistant-only loss masking."""

import pytest

from research.train.qwen_data import QwenMessage
from research.train.qwen_tokens import IGNORE_INDEX, tokenize_qwen_messages


class FakeChatTokenizer:
    """Small character tokenizer for offline assistant-mask tests."""

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render chat messages with the tokenizer chat template."""
        assert tokenize is False

        result = ""

        for message in conversation:
            result += f"<{message['role']}>{message['content']}</{message['role']}>"

        if add_generation_prompt:
            result += "<assistant>"

        return result

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ):
        """Run the callable interface for this object."""
        assert add_special_tokens is False
        assert return_offsets_mapping is True

        return {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _messages() -> tuple[QwenMessage, ...]:
    """Build the chat messages used by tokenizer tests."""
    return (
        QwenMessage(role="system", content="system"),
        QwenMessage(role="user", content="John"),
        QwenMessage(role="assistant", content='{"spans":[]}'),
    )


def test_masks_prompt_tokens() -> None:
    """System, user, and assistant prefix should not contribute to loss."""

    tokenizer = FakeChatTokenizer()
    messages = _messages()

    result = tokenize_qwen_messages(
        messages,
        tokenizer,
        max_length=512,
    )

    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert result["labels"][: len(prompt)] == [IGNORE_INDEX] * len(prompt)


def test_keeps_assistant_response_tokens() -> None:
    """Assistant response and terminator should remain visible to loss."""

    tokenizer = FakeChatTokenizer()
    messages = _messages()

    result = tokenize_qwen_messages(
        messages,
        tokenizer,
        max_length=512,
    )

    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert result["labels"][len(prompt) :] == result["input_ids"][len(prompt) :]


def test_input_and_label_lengths_match() -> None:
    """Training arrays should remain aligned."""

    result = tokenize_qwen_messages(
        _messages(),
        FakeChatTokenizer(),
        max_length=512,
    )

    assert len(result["input_ids"]) == len(result["attention_mask"])
    assert len(result["input_ids"]) == len(result["labels"])


def test_attention_mask_marks_every_real_token() -> None:
    """Unpadded records should mark every token as visible."""

    result = tokenize_qwen_messages(
        _messages(),
        FakeChatTokenizer(),
        max_length=512,
    )

    assert result["attention_mask"] == [1] * len(result["input_ids"])


def test_rejects_overlong_example() -> None:
    """Examples beyond context should fail rather than truncate targets."""

    with pytest.raises(ValueError, match="exceeding max_length"):
        tokenize_qwen_messages(
            _messages(),
            FakeChatTokenizer(),
            max_length=5,
        )


def test_rejects_wrong_role_order() -> None:
    """Training messages require system/user/assistant ordering."""

    messages = (
        QwenMessage(role="user", content="John"),
        QwenMessage(role="system", content="system"),
        QwenMessage(role="assistant", content='{"spans":[]}'),
    )

    with pytest.raises(ValueError, match="system, user, assistant"):
        tokenize_qwen_messages(
            messages,
            FakeChatTokenizer(),
            max_length=512,
        )


def test_rejects_missing_message() -> None:
    """Every example needs all three training messages."""

    with pytest.raises(ValueError, match="exactly three"):
        tokenize_qwen_messages(
            _messages()[:2],
            FakeChatTokenizer(),
            max_length=512,
        )


def test_rejects_non_positive_max_length() -> None:
    """Context length must be positive."""

    with pytest.raises(ValueError, match="positive"):
        tokenize_qwen_messages(
            _messages(),
            FakeChatTokenizer(),
            max_length=0,
        )


def test_rejects_template_text_prefix_mismatch() -> None:
    """Unexpected chat-template rendering should fail loudly."""

    class BrokenTokenizer(FakeChatTokenizer):
        """Provide a tokenizer stub that intentionally breaks the expected contract."""

        def apply_chat_template(
            self,
            conversation,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            """Render chat messages with the tokenizer chat template."""
            result = super().apply_chat_template(
                conversation,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

            if len(conversation) == 3:
                return f"BROKEN{result}"

            return result

    with pytest.raises(ValueError, match="not a prefix"):
        tokenize_qwen_messages(
            _messages(),
            BrokenTokenizer(),
            max_length=512,
        )
