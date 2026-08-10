"""Tests for Qwen LoRA Trainer configuration."""

from pathlib import Path
from unittest.mock import patch

import torch

from research.train.qwen import (
    QwenRecordDataset,
    _training_arguments,
    _training_dtype,
)
from research.train.qwen_dataset import QwenTrainingRecord


def _config() -> dict:
    return {
        "seed": 42,
        "learning_rate": 2.0e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "epochs": 3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "logging_steps": 25,
        "gradient_checkpointing": True,
    }


def test_record_dataset_preserves_records() -> None:
    """Torch wrapper should expose prepared records unchanged."""

    record: QwenTrainingRecord = {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
        "labels": [-100, 2],
    }

    dataset = QwenRecordDataset([record])

    assert len(dataset) == 1
    assert dataset[0] == record


def test_diagnostic_arguments_use_one_step(tmp_path: Path) -> None:
    """Diagnostic mode should perform only one optimizer update."""

    args = _training_arguments(
        _config(),
        output_dir=tmp_path,
        diagnostic=True,
        dtype=torch.float32,
    )

    assert args.max_steps == 1
    assert args.logging_steps == 1
    assert args.eval_steps == 1
    assert args.save_steps == 1


def test_diagnostic_disables_mixed_precision_on_cpu(
    tmp_path: Path,
) -> None:
    """Float32 configuration should not enable mixed precision."""

    args = _training_arguments(
        _config(),
        output_dir=tmp_path,
        diagnostic=True,
        dtype=torch.float32,
    )

    assert args.fp16 is False
    assert args.bf16 is False


def test_fp16_configuration(tmp_path: Path) -> None:
    """FP16 selection should propagate into Trainer arguments."""

    args = _training_arguments(
        _config(),
        output_dir=tmp_path,
        diagnostic=True,
        dtype=torch.float16,
    )

    assert args.fp16 is True
    assert args.bf16 is False


@patch(
    "research.train.qwen.torch.cuda.is_available",
    return_value=False,
)
def test_training_dtype_uses_float32_without_cuda(
    _cuda_available,
) -> None:
    """CPU execution should use float32."""

    assert _training_dtype() == torch.float32


@patch(
    "research.train.qwen.torch.cuda.get_device_capability",
    return_value=(6, 0),
)
@patch(
    "research.train.qwen.torch.cuda.is_available",
    return_value=True,
)
def test_training_dtype_uses_fp16_on_pascal(
    _cuda_available,
    _capability,
) -> None:
    """Pascal CUDA hardware should use float16."""

    assert _training_dtype() == torch.float16


@patch(
    "research.train.qwen.torch.cuda.get_device_capability",
    return_value=(8, 0),
)
@patch(
    "research.train.qwen.torch.cuda.is_bf16_supported",
    return_value=True,
)
@patch(
    "research.train.qwen.torch.cuda.is_available",
    return_value=True,
)
def test_training_dtype_uses_bf16_when_supported(
    _cuda_available,
    _bf16_supported,
    _capability,
) -> None:
    """Ampere-or-newer CUDA hardware should prefer bfloat16."""

    assert _training_dtype() == torch.bfloat16
