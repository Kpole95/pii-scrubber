"""Tests for Qwen inference helpers."""

from unittest.mock import MagicMock

import torch

from research.eval.qwen_inference import generate_qwen_output


class FakeBatch(dict):
    """Minimal BatchEncoding-like object for inference tests."""

    def to(self, _device: object) -> "FakeBatch":
        return self


def test_generate_qwen_output_uses_chat_template() -> None:
    """Inference should render the prompt and decode only generated tokens."""

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    inputs = FakeBatch(
        {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )

    tokenizer.apply_chat_template.return_value = inputs
    tokenizer.decode.return_value = '{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}'

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[10, 11, 12, 20, 21, 22]])

    output = generate_qwen_output(
        model,
        tokenizer,
        "John called.",
    )

    assert output == ('{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}')

    tokenizer.apply_chat_template.assert_called_once()
    model.generate.assert_called_once()

    call_kwargs = model.generate.call_args.kwargs

    assert torch.equal(
        call_kwargs["input_ids"],
        inputs["input_ids"],
    )
    assert torch.equal(
        call_kwargs["attention_mask"],
        inputs["attention_mask"],
    )
    assert call_kwargs["max_new_tokens"] == 256
    assert call_kwargs["do_sample"] is False
    assert call_kwargs["pad_token_id"] == 0
    assert call_kwargs["eos_token_id"] == 2

    tokenizer.decode.assert_called_once()

    generated_ids = tokenizer.decode.call_args.args[0]

    assert torch.equal(
        generated_ids,
        torch.tensor([20, 21, 22]),
    )
    assert tokenizer.decode.call_args.kwargs == {
        "skip_special_tokens": True,
    }


def test_generate_qwen_output_accepts_custom_token_limit() -> None:
    """Inference should allow evaluation to set a generation limit."""

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2

    tokenizer.apply_chat_template.return_value = FakeBatch(
        {
            "input_ids": torch.tensor([[10]]),
            "attention_mask": torch.tensor([[1]]),
        }
    )
    tokenizer.decode.return_value = '{"spans":[]}'

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[10, 20]])

    output = generate_qwen_output(
        model,
        tokenizer,
        "No PII here.",
        max_new_tokens=128,
    )

    assert output == '{"spans":[]}'
    assert model.generate.call_args.kwargs["max_new_tokens"] == 128
