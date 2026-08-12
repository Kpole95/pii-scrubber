"""Tests for complete normalized dataset artifacts."""

import json
from pathlib import Path

import pytest

from pii_scrub.text import CharacterSpan
from research.data.artifacts import (
    DATASET_ARTIFACT_VERSION,
    DatasetManifest,
    LoadedDatasetArtifact,
    dataset_manifest_from_dict,
    dataset_manifest_to_dict,
    load_dataset_split,
    save_dataset_split,
)
from research.data.models import DatasetExample
from research.data.splits import DatasetSplit


def _make_example(
    example_id: str,
) -> DatasetExample:
    """Create one normalized dataset example."""

    return DatasetExample(
        example_id=example_id,
        text=f"User {example_id}",
        spans=(
            CharacterSpan(
                start=0,
                end=4,
                entity_type="PERSON",
            ),
        ),
        source="test",
        language="en",
    )


def _make_split() -> DatasetSplit:
    """Create a small complete dataset split."""

    return DatasetSplit(
        train=(
            _make_example("train-1"),
            _make_example("train-2"),
        ),
        validation=(_make_example("validation-1"),),
        test=(_make_example("test-1"),),
    )


def test_save_dataset_split_creates_all_files(
    tmp_path: Path,
) -> None:
    """Saving a split should create three JSONL files and a manifest."""

    directory = tmp_path / "openpii"

    manifest = save_dataset_split(
        _make_split(),
        directory,
        source="openpii",
        seed=42,
    )

    assert (directory / "train.jsonl").is_file()
    assert (directory / "validation.jsonl").is_file()
    assert (directory / "test.jsonl").is_file()
    assert (directory / "manifest.json").is_file()

    assert manifest.source == "openpii"
    assert manifest.seed == 42
    assert manifest.train_count == 2
    assert manifest.validation_count == 1
    assert manifest.test_count == 1
    assert manifest.total_count == 4


def test_save_and_load_dataset_split_round_trip(
    tmp_path: Path,
) -> None:
    """A complete saved split should restore without changes."""

    original = _make_split()
    directory = tmp_path / "artifact"

    saved_manifest = save_dataset_split(
        original,
        directory,
        source="openpii",
        seed=99,
    )

    loaded = load_dataset_split(directory)

    assert loaded.split == original
    assert loaded.manifest == saved_manifest


def test_manifest_json_contains_expected_metadata(
    tmp_path: Path,
) -> None:
    """The manifest should contain counts, source, seed, and version."""

    directory = tmp_path / "artifact"

    save_dataset_split(
        _make_split(),
        directory,
        source="openpii",
        seed=42,
    )

    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert data == {
        "format_version": DATASET_ARTIFACT_VERSION,
        "source": "openpii",
        "seed": 42,
        "splits": {
            "train": 2,
            "validation": 1,
            "test": 1,
        },
        "total_count": 4,
    }


def test_dataset_manifest_round_trip() -> None:
    """Manifest mappings should restore the original model."""

    original = DatasetManifest(
        source="gretel_finance",
        seed=7,
        train_count=10,
        validation_count=2,
        test_count=3,
    )

    restored = dataset_manifest_from_dict(dataset_manifest_to_dict(original))

    assert restored == original


def test_load_dataset_split_rejects_count_mismatch(
    tmp_path: Path,
) -> None:
    """Loaded split sizes must agree with the manifest."""

    directory = tmp_path / "artifact"

    save_dataset_split(
        _make_split(),
        directory,
        source="openpii",
        seed=42,
    )

    manifest_path = directory / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["splits"]["train"] = 100
    data["total_count"] = 102

    manifest_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="train count does not match manifest",
    ):
        load_dataset_split(directory)


def test_load_dataset_split_rejects_unsupported_version(
    tmp_path: Path,
) -> None:
    """Unknown artifact versions should not be loaded silently."""

    directory = tmp_path / "artifact"

    save_dataset_split(
        _make_split(),
        directory,
        source="openpii",
        seed=42,
    )

    manifest_path = directory / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["format_version"] = 999

    manifest_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unsupported dataset artifact format version",
    ):
        load_dataset_split(directory)


def test_load_dataset_split_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """A saved artifact requires manifest.json."""

    directory = tmp_path / "artifact"
    directory.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="dataset manifest does not exist",
    ):
        load_dataset_split(directory)


def test_dataset_manifest_rejects_incorrect_total() -> None:
    """Manifest total_count must equal its split counts."""

    with pytest.raises(
        ValueError,
        match="total_count does not match split counts",
    ):
        dataset_manifest_from_dict(
            {
                "format_version": 1,
                "source": "openpii",
                "seed": 42,
                "splits": {
                    "train": 5,
                    "validation": 2,
                    "test": 1,
                },
                "total_count": 99,
            }
        )


def test_loaded_artifact_rejects_count_mismatch() -> None:
    """LoadedDatasetArtifact should validate its split counts."""

    split = _make_split()

    manifest = DatasetManifest(
        source="openpii",
        seed=42,
        train_count=99,
        validation_count=1,
        test_count=1,
    )

    with pytest.raises(
        ValueError,
        match="train count does not match manifest",
    ):
        LoadedDatasetArtifact(
            split=split,
            manifest=manifest,
        )


def test_save_dataset_split_rejects_empty_source(
    tmp_path: Path,
) -> None:
    """Saved artifacts require a meaningful source name."""

    with pytest.raises(
        ValueError,
        match="source must not be empty",
    ):
        save_dataset_split(
            _make_split(),
            tmp_path / "artifact",
            source=" ",
            seed=42,
        )
