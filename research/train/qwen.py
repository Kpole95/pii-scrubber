"""Train Qwen2.5 with LoRA for generative PII span extraction."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from research.data.artifacts import load_dataset_split
from research.train.qwen_collator import collate_qwen_records
from research.train.qwen_dataset import (
    QwenTrainingRecord,
    prepare_qwen_dataset,
)
from research.train.qwen_model import (
    load_qwen_lora_model,
    trainable_parameter_counts,
)
from research.train.qwen_tokens import ChatTokenizer


class QwenRecordDataset(Dataset[QwenTrainingRecord]):
    """Small torch Dataset wrapper around prepared Qwen records."""

    def __init__(self, records: list[QwenTrainingRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> QwenTrainingRecord:
        return self.records[index]


def main() -> None:
    """Load config and run either diagnostic or normal LoRA training."""

    args = _parser().parse_args()

    with open(args.config, encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    raw_tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_fast=True,
    )
    tokenizer = cast(ChatTokenizer, raw_tokenizer)

    if raw_tokenizer.pad_token_id is None:
        if raw_tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define EOS or padding token")
        raw_tokenizer.pad_token = raw_tokenizer.eos_token

    artifact = load_dataset_split(args.input)

    train_examples = artifact.split.train
    validation_examples = artifact.split.validation

    if args.diagnostic:
        train_examples = train_examples[:8]
        validation_examples = validation_examples[:2]

    train_records, train_overflow = prepare_qwen_dataset(
        train_examples,
        tokenizer,
        max_length=config["max_length"],
    )

    validation_records, validation_overflow = prepare_qwen_dataset(
        validation_examples,
        tokenizer,
        max_length=config["max_length"],
    )

    if not train_records:
        raise ValueError("no Qwen training records remain after preprocessing")

    if not validation_records:
        raise ValueError("no Qwen validation records remain after preprocessing")

    dtype = _training_dtype()

    model = load_qwen_lora_model(
        config["model_name"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        torch_dtype=dtype,
    )

    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainable, total = trainable_parameter_counts(model)

    print(f"Train examples:       {len(train_records):,}")
    print(f"Train overflow:       {train_overflow:,}")
    print(f"Validation examples:  {len(validation_records):,}")
    print(f"Validation overflow:  {validation_overflow:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Training dtype:       {dtype}")
    print(f"Diagnostic mode:      {args.diagnostic}")

    training_args = _training_arguments(
        config,
        output_dir=args.output,
        diagnostic=args.diagnostic,
        dtype=dtype,
    )

    pad_token_id = raw_tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer pad token ID must be configured")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=QwenRecordDataset(train_records),
        eval_dataset=QwenRecordDataset(validation_records),
        data_collator=lambda records: collate_qwen_records(
            records,
            pad_token_id=pad_token_id,
        ),
        processing_class=raw_tokenizer,
    )

    trainer.train()

    final_dir = args.output / "final"

    model.save_pretrained(final_dir)
    raw_tokenizer.save_pretrained(final_dir)

    print(f"Saved adapter: {final_dir}")


def _training_dtype() -> torch.dtype:
    """Choose a supported training dtype for the current hardware."""

    if not torch.cuda.is_available():
        return torch.float32

    major, _ = torch.cuda.get_device_capability()

    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def _training_arguments(
    config: dict[str, Any],
    *,
    output_dir: Path,
    diagnostic: bool,
    dtype: torch.dtype,
) -> TrainingArguments:
    """Build Trainer arguments from the frozen Stage 11 configuration."""

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "warmup_ratio": config["warmup_ratio"],
        "num_train_epochs": config["epochs"],
        "logging_strategy": "steps",
        "logging_steps": 1 if diagnostic else config["logging_steps"],
        "logging_first_step": True,
        "eval_strategy": "steps" if diagnostic else "epoch",
        "eval_steps": 1 if diagnostic else None,
        "save_strategy": "steps" if diagnostic else "epoch",
        "save_steps": 1 if diagnostic else 500,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": config["seed"],
        "data_seed": config["seed"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "fp16": dtype == torch.float16,
        "bf16": dtype == torch.bfloat16,
    }

    if diagnostic:
        kwargs["max_steps"] = 1

    return TrainingArguments(**kwargs)


def _parser() -> ArgumentParser:
    """Build the Qwen LoRA training CLI."""

    parser = ArgumentParser(
        description="Train Qwen2.5 with LoRA for PII detection.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen.yaml"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/data_artifacts/ai4privacy"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/qwen"),
    )

    parser.add_argument(
        "--diagnostic",
        action="store_true",
    )

    return parser


if __name__ == "__main__":
    main()
