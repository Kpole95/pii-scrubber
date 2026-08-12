"""Train the DeBERTa PII token classifier."""

import argparse
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from research.data.artifacts import load_dataset_split
from research.train.encoder_data import build_label_mapping, prepare_encoder_records


def main() -> None:
    """Train and save the encoder model."""
    args = _parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifact = load_dataset_split(args.data)

    train_examples = artifact.split.train
    validation_examples = artifact.split.validation

    if args.train_limit is not None:
        train_examples = train_examples[: args.train_limit]
    if args.validation_limit is not None:
        validation_examples = validation_examples[: args.validation_limit]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    labels = build_label_mapping(artifact.split.train)
    id2label = {index: label for label, index in labels.items()}

    train = prepare_encoder_records(
        train_examples,
        tokenizer,
        labels,
        strategy=config["label_strategy"],
        max_length=config["max_length"],
        overlap=config["window_overlap"],
    )
    validation = prepare_encoder_records(
        validation_examples,
        tokenizer,
        labels,
        strategy=config["label_strategy"],
        max_length=config["max_length"],
        overlap=config["window_overlap"],
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=len(labels),
        label2id=labels,
        id2label=id2label,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        seed=config["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        logging_nan_inf_filter=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(validation),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Train the encoder PII detector.")
    parser.add_argument("--config", type=Path, default=Path("configs/encoder.yaml"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("research/data_artifacts/ai4privacy"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/encoder"),
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--max-steps", type=int)
    return parser


if __name__ == "__main__":
    main()
