"""Tests for Qwen LoRA model helpers."""

from unittest.mock import MagicMock, patch

import pytest

from research.train.qwen_model import (
    load_qwen_lora_model,
    trainable_parameter_counts,
)


def test_rejects_non_positive_lora_rank() -> None:
    with pytest.raises(ValueError, match="lora_r must be positive"):
        load_qwen_lora_model(
            "fake-model",
            lora_r=0,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=("q_proj",),
        )


def test_rejects_non_positive_lora_alpha() -> None:
    with pytest.raises(ValueError, match="lora_alpha must be positive"):
        load_qwen_lora_model(
            "fake-model",
            lora_r=16,
            lora_alpha=0,
            lora_dropout=0.05,
            target_modules=("q_proj",),
        )


def test_rejects_invalid_lora_dropout() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_qwen_lora_model(
            "fake-model",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=1.0,
            target_modules=("q_proj",),
        )


def test_rejects_empty_target_modules() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        load_qwen_lora_model(
            "fake-model",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=(),
        )


@patch("research.train.qwen_model.get_peft_model")
@patch("research.train.qwen_model.AutoModelForCausalLM.from_pretrained")
def test_builds_lora_model(
    from_pretrained: MagicMock,
    get_peft_model: MagicMock,
) -> None:
    base_model = MagicMock()
    peft_model = MagicMock()

    from_pretrained.return_value = base_model
    get_peft_model.return_value = peft_model

    result = load_qwen_lora_model(
        "fake-model",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("q_proj", "v_proj"),
    )

    assert result is peft_model
    from_pretrained.assert_called_once_with("fake-model")
    get_peft_model.assert_called_once()

    config = get_peft_model.call_args.args[1]

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert set(config.target_modules) == {"q_proj", "v_proj"}


def test_counts_trainable_parameters() -> None:
    model = MagicMock()

    frozen = MagicMock()
    frozen.numel.return_value = 100
    frozen.requires_grad = False

    trainable = MagicMock()
    trainable.numel.return_value = 20
    trainable.requires_grad = True

    model.parameters.return_value = [frozen, trainable]

    assert trainable_parameter_counts(model) == (20, 120)
