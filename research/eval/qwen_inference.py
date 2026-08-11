"""Inference helpers for the Qwen LoRA PII detector."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from research.train.qwen_data import SYSTEM_PROMPT

QwenGenerator = Callable[[str], str]


def generate_qwen_output(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_new_tokens: int = 256,
) -> str:
    """Generate one raw JSON response from the Qwen PII detector."""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_length:]

    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return str(decoded).strip()


def load_qwen_generator(
    adapter_path: Path,
    *,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str | None = None,
    max_new_tokens: int = 256,
) -> QwenGenerator:
    """Load the Qwen base model and LoRA adapter for deterministic inference."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if resolved_device.startswith("cuda"):
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )
    model.to(resolved_device)
    model.eval()

    def generate(text: str) -> str:
        return generate_qwen_output(
            model,
            tokenizer,
            text,
            max_new_tokens=max_new_tokens,
        )

    return generate
