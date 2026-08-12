"""Immutable domain models shared by runtime components.

Offsets are always half-open and refer to the original input text. Keeping one
span representation prevents tokenizer-local offsets from leaking into public
APIs.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

Offset: TypeAlias = tuple[int, int]
TokenLabel: TypeAlias = str | None
SubwordLabelStrategy: TypeAlias = Literal["all_subwords", "first_subword"]


@dataclass(frozen=True, slots=True)
class CharacterSpan:
    """Represent one entity using half-open character offsets.

    Example:
        ``CharacterSpan(5, 11, "PERSON").extract("Call Murali.")`` returns
        ``"Murali"``.
    """

    start: int
    end: int
    entity_type: str

    def __post_init__(self) -> None:
        """Validate span offsets and the entity label."""
        for name, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"span {name} must be an integer")
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")
        if not isinstance(self.entity_type, str):
            raise TypeError("entity_type must be a string")
        if not self.entity_type.strip():
            raise ValueError("entity_type must not be empty")

    @property
    def length(self) -> int:
        """Return the number of characters covered by the span."""

        return self.end - self.start

    def extract(self, text: str) -> str:
        """Return the exact source substring covered by this span."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.end > len(text):
            raise ValueError(f"span end {self.end} exceeds text length {len(text)}")
        return text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class DetectedSpan(CharacterSpan):
    """A character span with an optional calibrated confidence score."""

    score: float | None = None

    def __post_init__(self) -> None:
        """Validate the detected span and optional confidence score."""
        super(DetectedSpan, self).__post_init__()
        if self.score is None:
            return
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("score must be a number or None")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be between 0 and 1")
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class WordOffset:
    """Store one source word and its exact character offsets."""

    word: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Validate one word and its source offsets."""
        if not isinstance(self.word, str):
            raise TypeError("word must be a string")
        if not self.word:
            raise ValueError("word must not be empty")
        for name, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"word {name} must be an integer")
        if self.start < 0:
            raise ValueError("word start must be non-negative")
        if self.end <= self.start:
            raise ValueError("word end must be greater than word start")
        if self.end - self.start != len(self.word):
            raise ValueError("word offset length must equal the source word length")

    def extract(self, text: str) -> str:
        """Extract the word and verify that the offsets still match it."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.end > len(text):
            raise ValueError(f"word end {self.end} exceeds text length {len(text)}")
        extracted = text[self.start : self.end]
        if extracted != self.word:
            raise ValueError(f"offset extracts {extracted!r}, expected {self.word!r}")
        return extracted


@dataclass(frozen=True, slots=True)
class AlignedExample:
    """Store token IDs, global offsets, word IDs, and aligned BIO labels."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offset_mapping: tuple[Offset, ...]
    word_ids: tuple[int | None, ...]
    token_labels: tuple[TokenLabel, ...]

    def __post_init__(self) -> None:
        """Validate aligned tokens, labels, and source offsets."""
        fields = (
            self.input_ids,
            self.attention_mask,
            self.offset_mapping,
            self.word_ids,
            self.token_labels,
        )
        if len({len(field) for field in fields}) != 1:
            raise ValueError("all AlignedExample fields must have equal lengths")
        if not self.input_ids:
            raise ValueError("AlignedExample must contain at least one token")
        for index, value in enumerate(self.input_ids):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"input ID at index {index} must be an integer")
        for index, value in enumerate(self.attention_mask):
            if value not in (0, 1):
                raise ValueError(f"attention mask at index {index} must be 0 or 1")
        for index, offset in enumerate(self.offset_mapping):
            if not _is_offset(offset):
                raise TypeError(f"offset at index {index} must contain two integers")
            if offset[0] < 0 or offset[1] < offset[0]:
                raise ValueError(f"offset at index {index} is invalid")
        for index, word_id in enumerate(self.word_ids):
            if word_id is not None and (isinstance(word_id, bool) or not isinstance(word_id, int)):
                raise TypeError(f"word ID at index {index} must be an integer or None")
            if word_id is not None and word_id < 0:
                raise ValueError(f"word ID at index {index} must be non-negative")
        for index, label in enumerate(self.token_labels):
            if label is not None and not isinstance(label, str):
                raise TypeError(f"token label at index {index} must be a string or None")
            if self.word_ids[index] is None and self.offset_mapping[index] != (0, 0):
                raise ValueError(f"special token at index {index} must use offset (0, 0)")
            if self.word_ids[index] is None and label is not None:
                raise ValueError(f"special token at index {index} must use label None")

    @property
    def token_count(self) -> int:
        """Return the number of tokenizer tokens."""

        return len(self.input_ids)


def _is_offset(value: object) -> bool:
    """Return whether a value is a valid character-offset pair."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(part, int) and not isinstance(part, bool) for part in value)
    )


@dataclass(frozen=True, slots=True)
class WindowSpan:
    """Store one document-global prediction produced by an input window."""

    window_index: int
    span: CharacterSpan
    score: float | None = None

    def __post_init__(self) -> None:
        """Validate one window-relative prediction."""
        if isinstance(self.window_index, bool) or not isinstance(self.window_index, int):
            raise TypeError("window_index must be an integer")
        if self.window_index < 0:
            raise ValueError("window_index must be non-negative")
        if not isinstance(self.span, CharacterSpan):
            raise TypeError("span must be a CharacterSpan")
        if self.score is None:
            return
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("score must be a number or None")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be between 0 and 1")
        object.__setattr__(self, "score", float(self.score))
