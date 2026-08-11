"""Inference helpers for the Qwen LoRA PII detector."""

from typing import Any

import torch

from research.train.qwen_data import SYSTEM_PROMPT


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
    