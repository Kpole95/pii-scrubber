"""Deterministic sampling and splitting for normalized datasets."""

import random
from dataclasses import dataclass

from research.data.models import DatasetExample


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Store deterministic train, validation, and test examples.

    Example:
        A dataset with ten records may be split into six training,
        two validation, and two test examples.
    """

    train: tuple[DatasetExample, ...]
    validation: tuple[DatasetExample, ...]
    test: tuple[DatasetExample, ...]

    def __post_init__(self) -> None:
        """Validate split containers and keep example IDs disjoint."""
        for field_name, examples in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if not isinstance(examples, tuple):
                raise TypeError(f"{field_name} must be a tuple")

            for index, example in enumerate(examples):
                if not isinstance(example, DatasetExample):
                    raise TypeError(
                        f"{field_name} example at index {index} must be a DatasetExample"
                    )

        all_ids = [
            example.example_id
            for example in (
                *self.train,
                *self.validation,
                *self.test,
            )
        ]

        if len(all_ids) != len(set(all_ids)):
            raise ValueError("dataset splits must not contain duplicate example IDs")

    @property
    def train_count(self) -> int:
        """Return the number of training examples."""

        return len(self.train)

    @property
    def validation_count(self) -> int:
        """Return the number of validation examples."""

        return len(self.validation)

    @property
    def test_count(self) -> int:
        """Return the number of test examples."""

        return len(self.test)

    @property
    def total_count(self) -> int:
        """Return the total number of split examples."""

        return self.train_count + self.validation_count + self.test_count


def sample_examples(
    examples: tuple[DatasetExample, ...],
    *,
    sample_size: int,
    seed: int = 42,
) -> tuple[DatasetExample, ...]:
    """Select a deterministic random subset.

    The original input tuple is not modified.

    Example:
        Calling this function twice with the same seed returns the
        same example IDs in the same order.
    """

    _validate_examples(examples)
    _validate_non_negative_integer(
        sample_size,
        field_name="sample_size",
    )
    _validate_seed(seed)

    if sample_size > len(examples):
        raise ValueError(f"sample_size {sample_size} exceeds dataset size {len(examples)}")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    return tuple(shuffled[:sample_size])


def split_examples(
    examples: tuple[DatasetExample, ...],
    *,
    validation_size: int,
    test_size: int,
    seed: int = 42,
) -> DatasetSplit:
    """Create deterministic train, validation, and test splits.

    The records are shuffled once. The first records become the test
    split, the next records become validation, and the rest become
    training data.
    """

    _validate_examples(examples)
    _validate_non_negative_integer(
        validation_size,
        field_name="validation_size",
    )
    _validate_non_negative_integer(
        test_size,
        field_name="test_size",
    )
    _validate_seed(seed)

    held_out_size = validation_size + test_size

    if held_out_size > len(examples):
        raise ValueError("validation_size plus test_size exceeds dataset size")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    test_end = test_size
    validation_end = test_end + validation_size

    test = tuple(shuffled[:test_end])
    validation = tuple(shuffled[test_end:validation_end])
    train = tuple(shuffled[validation_end:])

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )


def _validate_examples(
    examples: tuple[DatasetExample, ...],
) -> None:
    """Validate a normalized example collection."""

    if not isinstance(examples, tuple):
        raise TypeError("examples must be a tuple")

    example_ids: list[str] = []

    for index, example in enumerate(examples):
        if not isinstance(example, DatasetExample):
            raise TypeError(f"example at index {index} must be a DatasetExample")

        example_ids.append(example.example_id)

    if len(example_ids) != len(set(example_ids)):
        raise ValueError("examples must contain unique example IDs")


def _validate_non_negative_integer(
    value: int,
    *,
    field_name: str,
) -> None:
    """Validate a non-negative integer argument."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")

    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_seed(seed: int) -> None:
    """Validate the deterministic random seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
