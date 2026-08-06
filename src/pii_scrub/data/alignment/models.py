"""Immutable data models used by the alignment pipeline."""

from dataclasses import dataclass

from .types import Offset, TokenLabel


@dataclass(frozen=True, slots=True)
class CharacterSpan:
    """Represent one entity using half-open character offsets.

    ``start`` is included and ``end`` is excluded.

    Example:
        ``CharacterSpan(5, 11, "PERSON")`` extracts ``"Murali"``
        from ``"Call Murali."``.
    """

    start: int
    end: int
    entity_type: str

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("span start must be an integer")

        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("span end must be an integer")

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
        """Return the number of characters in the span."""

        return self.end - self.start

    def extract(self, text: str) -> str:
        """Extract this span from its original text.

        Example:
            ``CharacterSpan(5, 11, "PERSON").extract("Call Murali.")``
            returns ``"Murali"``.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if self.end > len(text):
            raise ValueError(
                f"span end {self.end} exceeds text length {len(text)}"
            )

        return text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class WordOffset:
    """Store one source word and its exact character offsets.

    Example:
        ``WordOffset("Murali", 5, 11)`` describes ``"Murali"`` in
        ``"Call Murali."``.
    """

    word: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.word, str):
            raise TypeError("word must be a string")

        if not self.word:
            raise ValueError("word must not be empty")

        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("word start must be an integer")

        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("word end must be an integer")

        if self.start < 0:
            raise ValueError("word start must be non-negative")

        if self.end <= self.start:
            raise ValueError("word end must be greater than word start")

        if self.end - self.start != len(self.word):
            raise ValueError(
                "word offset length must equal the source word length"
            )

    def extract(self, text: str) -> str:
        """Extract and validate the word from its original text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if self.end > len(text):
            raise ValueError(
                f"word end {self.end} exceeds text length {len(text)}"
            )

        extracted = text[self.start : self.end]

        if extracted != self.word:
            raise ValueError(
                f"offset extracts {extracted!r}, expected {self.word!r}"
            )

        return extracted


@dataclass(frozen=True, slots=True)
class AlignedExample:
    """Store one tokenized example and its aligned BIO labels.

    Every tuple contains one item per tokenizer token.

    Example:
        A special token uses ``word_id=None``, offset ``(0, 0)``,
        and label ``None``.
    """

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offset_mapping: tuple[Offset, ...]
    word_ids: tuple[int | None, ...]
    token_labels: tuple[TokenLabel, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.attention_mask),
            len(self.offset_mapping),
            len(self.word_ids),
            len(self.token_labels),
        }

        if len(lengths) != 1:
            raise ValueError(
                "all AlignedExample fields must have equal lengths"
            )

        if not self.input_ids:
            raise ValueError(
                "AlignedExample must contain at least one token"
            )

        for token_index, input_id in enumerate(self.input_ids):
            if isinstance(input_id, bool) or not isinstance(input_id, int):
                raise TypeError(
                    f"input ID at index {token_index} must be an integer"
                )

        for token_index, mask_value in enumerate(self.attention_mask):
            if mask_value not in (0, 1):
                raise ValueError(
                    f"attention mask at index {token_index} must be 0 or 1"
                )

        for token_index, offset in enumerate(self.offset_mapping):
            if (
                not isinstance(offset, tuple)
                or len(offset) != 2
                or isinstance(offset[0], bool)
                or isinstance(offset[1], bool)
                or not isinstance(offset[0], int)
                or not isinstance(offset[1], int)
            ):
                raise TypeError(
                    f"offset at index {token_index} must contain two integers"
                )

            start, end = offset

            if start < 0 or end < start:
                raise ValueError(
                    f"offset at index {token_index} is invalid"
                )

        for token_index, word_id in enumerate(self.word_ids):
            if word_id is not None:
                if isinstance(word_id, bool) or not isinstance(word_id, int):
                    raise TypeError(
                        f"word ID at index {token_index} "
                        "must be an integer or None"
                    )

                if word_id < 0:
                    raise ValueError(
                        f"word ID at index {token_index} "
                        "must be non-negative"
                    )

        for token_index, label in enumerate(self.token_labels):
            if label is not None and not isinstance(label, str):
                raise TypeError(
                    f"token label at index {token_index} "
                    "must be a string or None"
                )

            if self.word_ids[token_index] is None:
                if self.offset_mapping[token_index] != (0, 0):
                    raise ValueError(
                        f"special token at index {token_index} "
                        "must use offset (0, 0)"
                    )

                if label is not None:
                    raise ValueError(
                        f"special token at index {token_index} "
                        "must use label None"
                    )

    @property
    def token_count(self) -> int:
        """Return the number of tokenizer tokens."""

        return len(self.input_ids)