"""BIO-to-subword alignment utilities.

This module converts word-level BIO annotations into labels aligned with
fast-tokenizer subwords while preserving exact character offsets in the
original text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

from transformers import PreTrainedTokenizerFast
SubwordLabelStrategy = Literal[
    "all_subwords",
    "first_subword",
]

@dataclass(frozen=True, slots=True)
class CharacterSpan:
    """Represent one typed entity with half-open character offsets."""

    start: int
    end: int
    entity_type: str

    def __post_init__(self) -> None:
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
        if self.entity_type != self.entity_type.strip():
            raise ValueError("entity_type must not contain surrounding whitespace")

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
        return text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class WordOffset:
    """A source word and its half-open character offsets in the original text."""

    word: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.word, str):
            raise TypeError("word must be a string")
        if not self.word:
            raise ValueError("word must not be empty")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("end must be an integer")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.end - self.start != len(self.word):
            raise ValueError("word length must match the character-offset length")

    def extract(self, text: str) -> str:
        """Extract and validate this word against the original text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.end > len(text):
            raise ValueError(
                f"word end {self.end} exceeds text length {len(text)}"
            )

        extracted = text[self.start : self.end]
        if extracted != self.word:
            raise ValueError(
                f"offset text {extracted!r} does not match word {self.word!r}"
            )
        return extracted


@dataclass(frozen=True, slots=True)
class AlignedExample:
    """Store one tokenized example with aligned labels and global offsets."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    word_ids: tuple[int | None, ...]
    token_labels: tuple[str | None, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.attention_mask),
            len(self.offset_mapping),
            len(self.word_ids),
            len(self.token_labels),
        }
        if len(lengths) != 1:
            raise ValueError("all AlignedExample fields must have equal lengths")
        if not self.input_ids:
            raise ValueError("AlignedExample must contain at least one token")

        for index, (offset, word_id, label) in enumerate(
            zip(
                self.offset_mapping,
                self.word_ids,
                self.token_labels,
                strict=True,
            )
        ):
            if (
                not isinstance(offset, tuple)
                or len(offset) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in offset)
            ):
                raise TypeError(
                    f"offset_mapping[{index}] must be a two-integer tuple"
                )

            start, end = offset
            if word_id is None:
                if offset != (0, 0):
                    raise ValueError(
                        "special-token offsets must be represented as (0, 0)"
                    )
                if label is not None:
                    raise ValueError("special-token labels must be None")
            else:
                if isinstance(word_id, bool) or not isinstance(word_id, int):
                    raise TypeError("word IDs must be integers or None")

                if word_id < 0:
                    raise ValueError("word IDs must be non-negative")

                if start < 0 or end <= start:
                    raise ValueError("source-token offsets must be non-empty")

                if label is not None:
                    parse_bio_label(label)

    @property
    def token_count(self) -> int:
        """Return the number of encoded tokens, including special tokens."""

        return len(self.input_ids)



def parse_bio_label(label: str) -> tuple[str, str | None]:
    """Parse and validate one BIO label."""

    if not isinstance(label, str):
        raise TypeError("BIO label must be a string")

    if label == "O":
        return "O", None

    if label != label.strip():
        raise ValueError("BIO label must not contain surrounding whitespace")

    if "-" not in label:
        raise ValueError("BIO label must be 'O', 'B-<TYPE>', or 'I-<TYPE>'")

    prefix, entity_type = label.split("-", maxsplit=1)

    if prefix not in {"B", "I"}:
        raise ValueError("BIO prefix must be 'B', 'I', or 'O'")
    if not entity_type:
        raise ValueError("BIO entity type must not be empty")
    if entity_type != entity_type.strip():
        raise ValueError("BIO entity type must not contain surrounding whitespace")

    return prefix, entity_type



def locate_words(text: str, words: list[str]) -> list[WordOffset]:
    """Locate an ordered sequence of words in the original text.

    Each word is searched for after the end of the previous match. This
    preserves repeated-word positions and all original whitespace and
    punctuation between words.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(words, list):
        raise TypeError("words must be a list")

    offsets: list[WordOffset] = []
    search_start = 0

    for index, word in enumerate(words):
        if not isinstance(word, str):
            raise TypeError(f"word at index {index} must be a string")
        if not word:
            raise ValueError(f"word at index {index} must not be empty")

        word_start = text.find(word, search_start)
        if word_start == -1:
            raise ValueError(
                f"word {word!r} at index {index} was not found "
                f"after character {search_start}"
            )

        word_end = word_start + len(word)
        offset = WordOffset(word=word, start=word_start, end=word_end)
        offset.extract(text)
        offsets.append(offset)
        search_start = word_end

    return offsets



def bio_tags_to_spans(
    text: str,
    words: list[str],
    labels: list[str],
) -> list[CharacterSpan]:
    """Convert word-level BIO labels into character-level entity spans."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(words, list):
        raise TypeError("words must be a list")
    if not isinstance(labels, list):
        raise TypeError("labels must be a list")
    if len(words) != len(labels):
        raise ValueError("words and labels must contain the same number of items")

    parsed_labels = [parse_bio_label(label) for label in labels]
    word_offsets = locate_words(text, words)
    spans: list[CharacterSpan] = []

    active_start: int | None = None
    active_end: int | None = None
    active_entity_type: str | None = None

    def close_active_span() -> None:
        nonlocal active_start, active_end, active_entity_type

        if (
            active_start is not None
            and active_end is not None
            and active_entity_type is not None
        ):
            spans.append(
                CharacterSpan(
                    start=active_start,
                    end=active_end,
                    entity_type=active_entity_type,
                )
            )

        active_start = None
        active_end = None
        active_entity_type = None

    for index, (offset, parsed_label) in enumerate(
        zip(word_offsets, parsed_labels, strict=True)
    ):
        prefix, entity_type = parsed_label

        if prefix == "O":
            close_active_span()
            continue

        if prefix == "B":
            close_active_span()
            active_start = offset.start
            active_end = offset.end
            active_entity_type = entity_type
            continue

        # prefix == "I"
        if active_entity_type is None:
            raise ValueError(f"I-label at index {index} has no active entity")
        if entity_type != active_entity_type:
            raise ValueError(
                f"I-label at index {index} has type {entity_type!r}, "
                f"but the active entity has type {active_entity_type!r}"
            )

        active_end = offset.end

    close_active_span()
    return spans



def align_bio_to_subwords(
    text: str,
    words: list[str],
    labels: list[str],
    tokenizer: PreTrainedTokenizerFast,
    strategy: SubwordLabelStrategy = "all_subwords",
    max_length: int | None = None,
) -> AlignedExample:
    """Align word-level BIO labels with tokenizer subwords.

    Args:
        text: Original unsplit source text.
        words: Ordered annotated words.
        labels: One BIO label for every word.
        tokenizer: Fast Hugging Face tokenizer.
        strategy: Label all subwords or only the first subword.
        max_length: Optional maximum token count, including special tokens.

    Raises:
        ValueError: If tokenization exceeds ``max_length``.

    Example:
        max_length=128 rejects an encoding containing more than 128 tokens.
    """

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("tokenizer must be a fast Hugging Face tokenizer")

    if strategy not in {"all_subwords", "first_subword"}:
        raise ValueError(
            "strategy must be 'all_subwords' or 'first_subword'"
        )
    if max_length is not None:
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer or None")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
    if len(words) != len(labels):
        raise ValueError(
            "words and labels must contain the same number of items"
        )

    word_offsets = locate_words(text, words)
    parsed_labels = [parse_bio_label(label) for label in labels]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
        add_special_tokens=True,
        truncation=False,
    )

    token_count = len(encoding["input_ids"])

    if max_length is not None and token_count > max_length:
        raise ValueError(
            f"tokenized example contains {token_count} tokens, "
            f"which exceeds max_length={max_length}; "
            "window the input instead of truncating it"
        )

    word_ids = encoding.word_ids()
    tokens = encoding.tokens()

    if word_ids is None:
        raise ValueError("fast tokenizer did not return word IDs")

    local_offsets = encoding["offset_mapping"]

    global_offsets: list[tuple[int, int]] = []
    token_labels: list[str | None] = []
    seen_word_ids: set[int] = set()
    covered_ranges_by_word: dict[int, list[tuple[int, int]]] = {}

    previous_word_id: int | None = None

    for token_index, word_id in enumerate(word_ids):
        local_start, local_end = local_offsets[token_index]

        if word_id is None:
            global_offsets.append((0, 0))
            token_labels.append(None)
            previous_word_id = None
            continue

        if word_id >= len(words):
            raise ValueError(
                f"token refers to invalid word index {word_id}"
            )
        token_text = tokens[token_index]
        prefix, _ = parsed_labels[word_id]

        if (
            tokenizer.unk_token is not None
            and token_text == tokenizer.unk_token
            and prefix != "O"
        ):
            raise ValueError(
                f"annotated word {words[word_id]!r} at index {word_id} "
                "was converted to the unknown token"
            )

        word_offset = word_offsets[word_id]

        if local_start < 0 or local_end > len(word_offset.word):
            raise ValueError(
                f"token offset {(local_start, local_end)} is outside "
                f"word {word_offset.word!r}"
            )

        global_start = word_offset.start + local_start
        global_end = word_offset.start + local_end

        if global_end <= global_start:
            raise ValueError(
                f"token at index {token_index} has an empty offset"
            )

        global_offsets.append((global_start, global_end))
        seen_word_ids.add(word_id)

        covered_ranges_by_word.setdefault(word_id, []).append(
            (local_start, local_end)
        )

        prefix, entity_type = parsed_labels[word_id]
        is_first_subword = word_id != previous_word_id

        if prefix == "O":
            if strategy == "first_subword" and not is_first_subword:
                aligned_label = None
            else:
                aligned_label = "O"

        elif is_first_subword:
            aligned_label = labels[word_id]

        elif strategy == "first_subword":
            aligned_label = None

        else:
            aligned_label = f"I-{entity_type}"

        token_labels.append(aligned_label)
        previous_word_id = word_id

    missing_word_ids = set(range(len(words))) - seen_word_ids

    if missing_word_ids:
        missing_words = [
            words[index]
            for index in sorted(missing_word_ids)
        ]

        raise ValueError(
            f"tokenizer dropped source words: {missing_words}"
        )

    for word_id, word_offset in enumerate(word_offsets):
        ranges = sorted(covered_ranges_by_word.get(word_id, []))

        if not ranges:
            raise ValueError(
                f"tokenizer produced no offsets for word "
                f"{word_offset.word!r} at index {word_id}"
            )

        covered_start = ranges[0][0]
        covered_end = ranges[-1][1]

        if covered_start != 0 or covered_end != len(word_offset.word):
            raise ValueError(
                f"token offsets do not fully cover word "
                f"{word_offset.word!r} at index {word_id}"
            )

        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] != current[0]:
                raise ValueError(
                    f"token offsets contain a gap or overlap for word "
                    f"{word_offset.word!r} at index {word_id}"
                )

    return AlignedExample(
        input_ids=tuple(encoding["input_ids"]),
        attention_mask=tuple(encoding["attention_mask"]),
        offset_mapping=tuple(global_offsets),
        word_ids=tuple(word_ids),
        token_labels=tuple(token_labels),
    )

def labels_to_ids(
    token_labels: tuple[str | None, ...],
    label_to_id: dict[str, int],
    ignore_index: int = -100,
) -> tuple[int, ...]:
    """Convert aligned BIO labels into integer model-training labels.

    ``None`` labels become ``ignore_index`` so special tokens and ignored
    subwords do not contribute to the training loss.

    Example:
        labels_to_ids(
            ("O", "B-PERSON", None, "I-PERSON"),
            {"O": 0, "B-PERSON": 1, "I-PERSON": 2},
        )

        returns:
            (0, 1, -100, 2)
    """

    if not isinstance(token_labels, tuple):
        raise TypeError("token_labels must be a tuple")

    if not isinstance(label_to_id, dict):
        raise TypeError("label_to_id must be a dictionary")

    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")

    if not label_to_id:
        raise ValueError("label_to_id must not be empty")

    for label, label_id in label_to_id.items():
        parse_bio_label(label)

        if isinstance(label_id, bool) or not isinstance(label_id, int):
            raise TypeError(
                f"label ID for {label!r} must be an integer"
            )

        if label_id == ignore_index:
            raise ValueError(
                f"label ID for {label!r} must not equal ignore_index"
            )

    if len(set(label_to_id.values())) != len(label_to_id):
        raise ValueError("label IDs must be unique")

    numeric_labels: list[int] = []

    for index, label in enumerate(token_labels):
        if label is None:
            numeric_labels.append(ignore_index)
            continue

        parse_bio_label(label)

        if label not in label_to_id:
            raise ValueError(
                f"label {label!r} at token index {index} "
                "is missing from label_to_id"
            )

        numeric_labels.append(label_to_id[label])

    return tuple(numeric_labels)

def aligned_labels_to_spans(
    example: AlignedExample,
) -> list[CharacterSpan]:
    """Reconstruct character spans from aligned token-level BIO labels.

    Ignored continuation subwords still extend the active entity when they
    belong to the same source word.

    Example:
        Mur      -> B-PERSON, offset (5, 8), word_id 1
        ##ali    -> None,     offset (8, 11), word_id 1

        Returns:
            CharacterSpan(5, 11, "PERSON")
    """

    if not isinstance(example, AlignedExample):
        raise TypeError("example must be an AlignedExample")

    spans: list[CharacterSpan] = []

    active_start: int | None = None
    active_end: int | None = None
    active_entity_type: str | None = None
    active_word_id: int | None = None

    def close_active_span() -> None:
        """Store the active entity and clear its temporary state."""

        nonlocal active_start
        nonlocal active_end
        nonlocal active_entity_type
        nonlocal active_word_id

        if (
            active_start is not None
            and active_end is not None
            and active_entity_type is not None
        ):
            spans.append(
                CharacterSpan(
                    start=active_start,
                    end=active_end,
                    entity_type=active_entity_type,
                )
            )

        active_start = None
        active_end = None
        active_entity_type = None
        active_word_id = None

    for token_index, (offset, word_id, label) in enumerate(
        zip(
            example.offset_mapping,
            example.word_ids,
            example.token_labels,
            strict=True,
        )
    ):
        start, end = offset

        if label is None:
            if (
                active_entity_type is not None
                and word_id is not None
                and word_id == active_word_id
            ):
                active_end = end

            continue

        prefix, entity_type = parse_bio_label(label)

        if prefix == "O":
            close_active_span()
            continue

        if prefix == "B":
            close_active_span()

            active_start = start
            active_end = end
            active_entity_type = entity_type
            active_word_id = word_id
            continue

        if active_entity_type is None:
            raise ValueError(
                f"I-label at token index {token_index} "
                "has no active entity"
            )

        if entity_type != active_entity_type:
            raise ValueError(
                f"I-label at token index {token_index} has type "
                f"{entity_type!r}, but active entity has type "
                f"{active_entity_type!r}"
            )

        active_end = end
        active_word_id = word_id

    close_active_span()

    return spans