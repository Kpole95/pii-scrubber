"""Internal data models shared by every dataset loader."""

from dataclasses import dataclass
from itertools import pairwise

from pii_scrub.types import CharacterSpan


@dataclass(frozen=True, slots=True)
class DatasetExample:
    """Represent one normalized PII-detection example.

    Every public dataset is converted into this model before training
    or evaluation.

    Example:
        ``DatasetExample(
            example_id="openpii-123",
            text="Call Murali.",
            spans=(CharacterSpan(5, 11, "PERSON"),),
            source="openpii",
            language="en",
        )``
    """

    example_id: str
    text: str
    spans: tuple[CharacterSpan, ...]
    source: str
    language: str

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str):
            raise TypeError("example_id must be a string")

        if not self.example_id.strip():
            raise ValueError("example_id must not be empty")

        if not isinstance(self.text, str):
            raise TypeError("text must be a string")

        if not self.text:
            raise ValueError("text must not be empty")

        if not isinstance(self.spans, tuple):
            raise TypeError("spans must be a tuple")

        for span_index, span in enumerate(self.spans):
            if not isinstance(span, CharacterSpan):
                raise TypeError(f"span at index {span_index} must be a CharacterSpan")

            if span.end > len(self.text):
                raise ValueError(
                    f"span at index {span_index} ends at "
                    f"{span.end}, beyond text length {len(self.text)}"
                )

            span.extract(self.text)

        if not isinstance(self.source, str):
            raise TypeError("source must be a string")

        if not self.source.strip():
            raise ValueError("source must not be empty")

        if not isinstance(self.language, str):
            raise TypeError("language must be a string")

        if not self.language.strip():
            raise ValueError("language must not be empty")

        sorted_spans = tuple(
            sorted(
                self.spans,
                key=lambda span: (
                    span.start,
                    span.end,
                    span.entity_type,
                ),
            )
        )

        if sorted_spans != self.spans:
            raise ValueError("spans must be sorted by character position")

        for previous, current in pairwise(self.spans):
            if current.start < previous.end:
                raise ValueError("DatasetExample spans must not overlap")

    @property
    def entity_count(self) -> int:
        """Return the number of annotated entities."""

        return len(self.spans)

    @property
    def entity_types(self) -> frozenset[str]:
        """Return the distinct normalized entity types.

        Example:
            PERSON, PERSON, EMAIL becomes ``{"PERSON", "EMAIL"}``.
        """

        return frozenset(span.entity_type for span in self.spans)


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """Describe one raw record that could not be normalized.

    Example:
        Record 4 is rejected because an annotation extends beyond the text.
    """

    record_index: int
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if isinstance(self.record_index, bool) or not isinstance(self.record_index, int):
            raise TypeError("record_index must be an integer")

        if self.record_index < 0:
            raise ValueError("record_index must be non-negative")

        if not isinstance(self.error_type, str):
            raise TypeError("error_type must be a string")

        if not self.error_type.strip():
            raise ValueError("error_type must not be empty")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string")

        if not self.message.strip():
            raise ValueError("message must not be empty")


@dataclass(frozen=True, slots=True)
class DatasetLoadReport:
    """Store accepted examples and rejected-record details.

    Example:
        Ten raw records produce eight examples and two rejections.
    """

    examples: tuple[DatasetExample, ...]
    rejected: tuple[RejectedRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.examples, tuple):
            raise TypeError("examples must be a tuple")

        if not isinstance(self.rejected, tuple):
            raise TypeError("rejected must be a tuple")

        for index, example in enumerate(self.examples):
            if not isinstance(example, DatasetExample):
                raise TypeError(f"example at index {index} must be a DatasetExample")

        for index, rejected_record in enumerate(self.rejected):
            if not isinstance(rejected_record, RejectedRecord):
                raise TypeError(f"rejected item at index {index} must be a RejectedRecord")

    @property
    def accepted_count(self) -> int:
        """Return the number of valid normalized examples."""

        return len(self.examples)

    @property
    def rejected_count(self) -> int:
        """Return the number of rejected raw records."""

        return len(self.rejected)

    @property
    def total_count(self) -> int:
        """Return the total number of processed records."""

        return self.accepted_count + self.rejected_count

    @property
    def acceptance_rate(self) -> float:
        """Return the fraction of records that were accepted."""

        if self.total_count == 0:
            return 0.0

        return self.accepted_count / self.total_count
