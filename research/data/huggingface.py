"""Load raw records from Hugging Face datasets."""

from collections.abc import Mapping
from typing import Any


def load_huggingface_records(
    dataset_id: str,
    *,
    split: str,
    config: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load one Hugging Face split as plain records.

    Example:
        ``load_huggingface_records("owner/data", split="test")``.
    """
    _validate_text(
        dataset_id,
        "dataset_id",
    )
    _validate_text(
        split,
        "split",
    )

    if config is not None:
        _validate_text(
            config,
            "config",
        )

    from datasets import load_dataset

    dataset = load_dataset(
        dataset_id,
        config,
        split=split,
    )

    records: list[dict[str, Any]] = []

    for index, record in enumerate(dataset):
        if not isinstance(record, Mapping):
            raise TypeError(f"record at index {index} must be a mapping")

        records.append(dict(record))

    return tuple(records)


def _validate_text(
    value: str,
    name: str,
) -> None:
    """Require a non-empty string.

    Example:
        ``"test"`` is valid; an empty string is not.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    if not value.strip():
        raise ValueError(f"{name} must not be empty")
