"""Immutable models for persisted dataset split artifacts."""

from dataclasses import dataclass

from research.data.splits import DatasetSplit

DATASET_ARTIFACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Describe the source, seed, version, and counts of one saved split."""

    source: str
    seed: int
    train_count: int
    validation_count: int
    test_count: int
    format_version: int = DATASET_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        """Validate manifest metadata and split counts."""
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name, count in (
            ("train_count", self.train_count),
            ("validation_count", self.validation_count),
            ("test_count", self.test_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{name} must be an integer")
            if count < 0:
                raise ValueError(f"{name} must be non-negative")
        if isinstance(self.format_version, bool) or not isinstance(self.format_version, int):
            raise TypeError("format_version must be an integer")
        if self.format_version <= 0:
            raise ValueError("format_version must be positive")

    @property
    def total_count(self) -> int:
        """Return the total number of examples across all splits."""

        return self.train_count + self.validation_count + self.test_count


@dataclass(frozen=True, slots=True)
class LoadedDatasetArtifact:
    """Pair a restored dataset split with its validated manifest."""

    split: DatasetSplit
    manifest: DatasetManifest

    def __post_init__(self) -> None:
        """Validate the loaded split against its manifest."""
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")
        if not isinstance(self.manifest, DatasetManifest):
            raise TypeError("manifest must be a DatasetManifest")
        for name in ("train", "validation", "test"):
            if getattr(self.split, f"{name}_count") != getattr(self.manifest, f"{name}_count"):
                raise ValueError(f"{name} count does not match manifest")
