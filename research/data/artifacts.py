"""Persist complete normalized dataset splits with a versioned manifest."""

import json
from pathlib import Path
from typing import Any

from research.data.artifact_models import (
    DATASET_ARTIFACT_VERSION,
    DatasetManifest,
    LoadedDatasetArtifact,
)
from research.data.serialization import read_dataset_jsonl, write_dataset_jsonl
from research.data.splits import DatasetSplit


def save_dataset_split(
    split: DatasetSplit,
    directory: str | Path,
    *,
    source: str,
    seed: int,
) -> DatasetManifest:
    """Write train, validation, test JSONL files and ``manifest.json``."""

    if not isinstance(split, DatasetSplit):
        raise TypeError("split must be a DatasetSplit")
    output = _directory(directory)
    manifest = DatasetManifest(
        source=source,
        seed=seed,
        train_count=split.train_count,
        validation_count=split.validation_count,
        test_count=split.test_count,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        write_dataset_jsonl(getattr(split, name), output / f"{name}.jsonl")
    (output / "manifest.json").write_text(
        json.dumps(dataset_manifest_to_dict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_dataset_split(directory: str | Path) -> LoadedDatasetArtifact:
    """Restore a split and reject missing, stale, or inconsistent artifacts."""

    root = _directory(directory)
    if not root.exists():
        raise FileNotFoundError(f"dataset artifact directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"dataset artifact path is not a directory: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest path is not a file: {manifest_path}")
    try:
        manifest = dataset_manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError("dataset manifest contains invalid JSON") from error
    if manifest.format_version != DATASET_ARTIFACT_VERSION:
        raise ValueError(f"unsupported dataset artifact format version: {manifest.format_version}")
    split = DatasetSplit(
        train=read_dataset_jsonl(root / "train.jsonl"),
        validation=read_dataset_jsonl(root / "validation.jsonl"),
        test=read_dataset_jsonl(root / "test.jsonl"),
    )
    return LoadedDatasetArtifact(split, manifest)


def dataset_manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    """Convert a validated manifest to JSON-compatible data."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    return {
        "format_version": manifest.format_version,
        "source": manifest.source,
        "seed": manifest.seed,
        "splits": {
            "train": manifest.train_count,
            "validation": manifest.validation_count,
            "test": manifest.test_count,
        },
        "total_count": manifest.total_count,
    }


def dataset_manifest_from_dict(data: object) -> DatasetManifest:
    """Validate decoded manifest JSON and create its immutable model."""

    if not isinstance(data, dict):
        raise TypeError("manifest data must be a mapping")
    splits = data.get("splits")
    if splits is None:
        raise ValueError("manifest is missing required field 'splits'")
    if not isinstance(splits, dict):
        raise TypeError("manifest field 'splits' must be a mapping")
    manifest = DatasetManifest(
        format_version=_required_int(data, "format_version", "manifest"),
        source=_required_text(data, "source", "manifest"),
        seed=_required_int(data, "seed", "manifest"),
        train_count=_required_int(splits, "train", "manifest splits"),
        validation_count=_required_int(splits, "validation", "manifest splits"),
        test_count=_required_int(splits, "test", "manifest splits"),
    )
    total = _required_int(data, "total_count", "manifest")
    if total != manifest.total_count:
        raise ValueError("manifest total_count does not match split counts")
    return manifest


def _directory(value: str | Path) -> Path:
    """Return the normalized artifact directory path."""
    if not isinstance(value, str | Path):
        raise TypeError("directory must be a string or Path")
    path = Path(value)
    if not str(path).strip():
        raise ValueError("directory must not be empty")
    return path


def _required_text(data: dict[str, Any], field: str, context: str) -> str:
    """Read and validate one required text field."""
    if field not in data:
        raise ValueError(f"{context} is missing required field {field!r}")
    value = data[field]
    if not isinstance(value, str):
        raise TypeError(f"{context} field {field!r} must be a string")
    if not value.strip():
        raise ValueError(f"{context} field {field!r} must not be empty")
    return value


def _required_int(data: dict[str, Any], field: str, context: str) -> int:
    """Read and validate one required integer field."""
    if field not in data:
        raise ValueError(f"{context} is missing required field {field!r}")
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} field {field!r} must be an integer")
    return value


__all__ = [
    "DATASET_ARTIFACT_VERSION",
    "DatasetManifest",
    "LoadedDatasetArtifact",
    "dataset_manifest_from_dict",
    "dataset_manifest_to_dict",
    "load_dataset_split",
    "save_dataset_split",
]
