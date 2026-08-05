"""BIO-to-subword alignment utilities.

This module converts word-level BIO annotations into labels aligned with
tokenizer subwords while preserving exact character offsets in the original
text.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CharacterSpan:
    """A typed half-open character span in the original text.

    The start position is inclusive and the end position is exclusive.
    Therefore, the represented text is obtained with ``text[start:end]``.
    """

    start: int
    end: int
    entity_type: str

    def __post_init__(self) -> None:
        """Validate the span immediately after construction."""

        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("start must be an integer")

        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("end must be an integer")

        if self.start < 0:
            raise ValueError("start must be non-negative")

        if self.end <= self.start:
            raise ValueError("end must be greater than start")

        if not isinstance(self.entity_type, str):
            raise TypeError("entity_type must be a string")

        if not self.entity_type.strip():
            raise ValueError("entity_type must not be empty")

    @property
    def length(self) -> int:
        """Return the number of characters covered by the span."""

        return self.end - self.start

    def extract(self, text: str) -> str:
        """Extract this span from its original text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if self.end > len(text):
            raise ValueError(
                f"span end {self.end} exceeds text length {len(text)}"
            )

        return text[self.start:self.end]


def parse_bio_label(label: str) -> tuple[str, str | None]:
    """Parse and validate one BIO label.

    Returns:
        A two-item tuple containing the BIO prefix and entity type.
        The entity type is ``None`` when the label is ``"O"``.

    Examples:
        ``"O"`` becomes ``("O", None)``.
        ``"B-PERSON"`` becomes ``("B", "PERSON")``.
        ``"I-EMAIL"`` becomes ``("I", "EMAIL")``.
    """

    if not isinstance(label, str):
        raise TypeError("BIO label must be a string")

    if label == "O":
        return "O", None

    if label != label.strip():
        raise ValueError(
            "BIO label must not contain surrounding whitespace"
        )

    if "-" not in label:
        raise ValueError(
            "BIO label must be 'O', 'B-<TYPE>', or 'I-<TYPE>'"
        )

    prefix, entity_type = label.split("-", maxsplit=1)

    if prefix not in {"B", "I"}:
        raise ValueError("BIO prefix must be 'B', 'I', or 'O'")

    if not entity_type:
        raise ValueError("BIO entity type must not be empty")

    if entity_type != entity_type.strip():
        raise ValueError(
            "BIO entity type must not contain surrounding whitespace"
        )

    return prefix, entity_type