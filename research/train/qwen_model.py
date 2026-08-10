"""Build the Qwen causal LM with its Stage 11 LoRA adapter."""

from collections.abc import Sequence
from typing import Any

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


def load_qwen_lora_model(
    model_name: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    torch_dtype: Any = None,
) -> Any:
    """Load a causal LM and attach the configured LoRA adapter."""

    if lora_r <= 0:
        raise ValueError("lora_r must be positive")
    if lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be between 0 and 1")
    if not target_modules:
        raise ValueError("target_modules must not be empty")

    load_kwargs: dict[str, Any] = {}

    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs,
    )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )

    return get_peft_model(model, config)


def trainable_parameter_counts(model: Any) -> tuple[int, int]:
    """Return trainable and total model parameter counts."""

    trainable = 0
    total = 0

    for parameter in model.parameters():
        count = parameter.numel()
        total += count

        if parameter.requires_grad:
            trainable += count

    return trainable, total
