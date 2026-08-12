"""Inspect the Stage 11 Qwen LoRA parameter footprint."""

from argparse import ArgumentParser

import yaml

from research.train.qwen_model import (
    load_qwen_lora_model,
    trainable_parameter_counts,
)


def main() -> None:
    """Load Qwen with LoRA and report trainable parameters."""

    args = _parser().parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model = load_qwen_lora_model(
        config["model_name"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
    )

    trainable, total = trainable_parameter_counts(model)

    print(f"Model: {config['model_name']}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.4f}%")

    targeted = getattr(model, "targeted_module_names", None)

    if targeted is not None:
        print(f"Targeted modules: {len(targeted)}")
        for name in targeted[:20]:
            print(f"  {name}")
        if len(targeted) > 20:
            print(f"  ... {len(targeted) - 20} more")


def _parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/qwen.yaml",
    )
    return parser


if __name__ == "__main__":
    main()
