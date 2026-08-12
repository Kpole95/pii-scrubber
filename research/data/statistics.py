"""Calculate summary statistics for normalized PII datasets."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from research.data.models import DatasetExample


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Describe the contents of a normalized dataset.

    Example:
        A dataset can contain 100 examples, 70 positive examples,
        30 negative examples, and 150 annotated entities.
    """

    example_count: int
    positive_example_count: int
    negative_example_count: int
    entity_count: int
    total_character_count: int
    minimum_text_length: int
    maximum_text_length: int
    average_text_length: float
    entity_type_counts: Mapping[str, int]
    source_counts: Mapping[str, int]
    language_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate aggregate counts and language statistics."""
        integer_fields = (
            ("example_count", self.example_count),
            (
                "positive_example_count",
                self.positive_example_count,
            ),
            (
                "negative_example_count",
                self.negative_example_count,
            ),
            ("entity_count", self.entity_count),
            (
                "total_character_count",
                self.total_character_count,
            ),
            (
                "minimum_text_length",
                self.minimum_text_length,
            ),
            (
                "maximum_text_length",
                self.maximum_text_length,
            ),
        )

        for field_name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")

            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if isinstance(self.average_text_length, bool) or not isinstance(
            self.average_text_length,
            int | float,
        ):
            raise TypeError("average_text_length must be numeric")

        if self.average_text_length < 0:
            raise ValueError("average_text_length must be non-negative")

        if self.positive_example_count + self.negative_example_count != self.example_count:
            raise ValueError("positive and negative counts must equal example_count")

        if self.example_count == 0:
            if (
                self.minimum_text_length != 0
                or self.maximum_text_length != 0
                or self.average_text_length != 0.0
            ):
                raise ValueError("empty dataset text statistics must be zero")
        elif self.minimum_text_length > self.maximum_text_length:
            raise ValueError("minimum_text_length cannot exceed maximum_text_length")

        for field_name, counts in (
            ("entity_type_counts", self.entity_type_counts),
            ("source_counts", self.source_counts),
            ("language_counts", self.language_counts),
        ):
            _validate_count_mapping(
                counts,
                field_name=field_name,
            )

    @property
    def positive_rate(self) -> float:
        """Return the fraction of examples containing PII."""

        if self.example_count == 0:
            return 0.0

        return self.positive_example_count / self.example_count

    @property
    def entities_per_example(self) -> float:
        """Return the average number of entities per example."""

        if self.example_count == 0:
            return 0.0

        return self.entity_count / self.example_count


def calculate_dataset_statistics(
    examples: tuple[DatasetExample, ...],
) -> DatasetStatistics:
    """Calculate summary statistics for normalized examples.

    The input must contain unique example IDs so one record cannot be
    accidentally counted more than once.
    """

    if not isinstance(examples, tuple):
        raise TypeError("examples must be a tuple")

    example_ids: list[str] = []
    text_lengths: list[int] = []

    positive_example_count = 0
    entity_count = 0
    entity_type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()

    for example_index, example in enumerate(examples):
        if not isinstance(example, DatasetExample):
            raise TypeError(f"example at index {example_index} must be a DatasetExample")

        example_ids.append(example.example_id)
        text_lengths.append(len(example.text))

        source_counts[example.source] += 1
        language_counts[example.language] += 1

        if example.spans:
            positive_example_count += 1

        entity_count += example.entity_count

        for span in example.spans:
            entity_type_counts[span.entity_type] += 1

    if len(example_ids) != len(set(example_ids)):
        raise ValueError("examples must contain unique example IDs")

    example_count = len(examples)
    negative_example_count = example_count - positive_example_count
    total_character_count = sum(text_lengths)

    if text_lengths:
        minimum_text_length = min(text_lengths)
        maximum_text_length = max(text_lengths)
        average_text_length = total_character_count / example_count
    else:
        minimum_text_length = 0
        maximum_text_length = 0
        average_text_length = 0.0

    return DatasetStatistics(
        example_count=example_count,
        positive_example_count=positive_example_count,
        negative_example_count=negative_example_count,
        entity_count=entity_count,
        total_character_count=total_character_count,
        minimum_text_length=minimum_text_length,
        maximum_text_length=maximum_text_length,
        average_text_length=average_text_length,
        entity_type_counts=dict(sorted(entity_type_counts.items())),
        source_counts=dict(sorted(source_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
    )


def _validate_count_mapping(
    counts: Mapping[str, int],
    *,
    field_name: str,
) -> None:
    """Validate a string-to-count mapping."""

    if not isinstance(counts, Mapping):
        raise TypeError(f"{field_name} must be a mapping")

    for key, count in counts.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")

        if not key.strip():
            raise ValueError(f"{field_name} keys must not be empty")

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{field_name} values must be integers")

        if count < 0:
            raise ValueError(f"{field_name} values must be non-negative")
