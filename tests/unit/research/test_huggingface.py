"""Tests for Hugging Face dataset acquisition."""

from typing import Any

import pytest

from research.data.huggingface import (
    load_huggingface_records,
)


def test_loads_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset rows should become plain dictionaries.

    Example:
        One source row becomes one returned dictionary.
    """

    def fake_load_dataset(
        dataset_id: str,
        config: str | None,
        *,
        split: str,
    ) -> list[dict[str, Any]]:
        assert dataset_id == "owner/data"
        assert config is None
        assert split == "test"

        return [
            {
                "id": "one",
                "text": "hello",
            }
        ]

    monkeypatch.setattr(
        "datasets.load_dataset",
        fake_load_dataset,
    )

    records = load_huggingface_records(
        "owner/data",
        split="test",
    )

    assert records == (
        {
            "id": "one",
            "text": "hello",
        },
    )


def test_passes_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optional dataset config should reach Hugging Face."""

    def fake_load_dataset(
        dataset_id: str,
        config: str | None,
        *,
        split: str,
    ) -> list[dict[str, Any]]:
        assert config == "english"
        return []

    monkeypatch.setattr(
        "datasets.load_dataset",
        fake_load_dataset,
    )

    records = load_huggingface_records(
        "owner/data",
        split="train",
        config="english",
    )

    assert records == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", ""),
        ("split", ""),
    ],
)
def test_rejects_empty_inputs(
    field: str,
    value: str,
) -> None:
    """Required Hugging Face identifiers must not be empty."""
    arguments = {
        "dataset_id": "owner/data",
        "split": "test",
    }
    arguments[field] = value

    with pytest.raises(
        ValueError,
        match=field,
    ):
        load_huggingface_records(
            arguments["dataset_id"],
            split=arguments["split"],
        )
