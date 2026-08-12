"""Align word-level BIO labels with tokenizer subwords.

The tokenizer is accepted through a small structural protocol rather than a
Transformers concrete class. Production callers can pass a Hugging Face fast
tokenizer, while tests can use a deterministic local fake without downloading
models.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pii_scrub.text.spans import parse_bio_label
from pii_scrub.types import AlignedExample, SubwordLabelStrategy, TokenLabel, WordOffset

IGNORE_INDEX = -100


class TokenEncoding(Protocol):
    """Minimal encoding interface required from a fast tokenizer result."""

    def __getitem__(self, key: str) -> Any:
        """Return one stored item by key."""

        ...

    def word_ids(self) -> list[int | None] | None:
        """Return tokenizer word identifiers for this encoding."""

        ...

    def tokens(self) -> list[str]:
        """Return token strings for this encoding."""

        ...


class FastTokenizer(Protocol):
    """Structural interface used by :func:`align_bio_to_subwords`."""

    unk_token: str | None

    def __call__(self, words: list[str], **kwargs: Any) -> TokenEncoding:
        """Tokenize pre-split words and return encoding metadata."""

        ...


def locate_words(text: str, words: Sequence[str]) -> list[WordOffset]:
    """Locate annotated words sequentially in the original text.

    Sequential search preserves repeated words, irregular whitespace, and
    Unicode offsets. For example, the two ``John`` values in
    ``"John called John."`` receive different offsets.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(words, Sequence) or isinstance(words, str):
        raise TypeError("words must be a sequence of strings")

    offsets: list[WordOffset] = []
    search_start = 0
    for index, word in enumerate(words):
        if not isinstance(word, str):
            raise TypeError(f"word at index {index} must be a string")
        if not word:
            raise ValueError(f"word at index {index} must not be empty")
        start = text.find(word, search_start)
        if start == -1:
            raise ValueError(
                f"word {word!r} at index {index} was not found after character {search_start}"
            )
        offset = WordOffset(word=word, start=start, end=start + len(word))
        offset.extract(text)
        offsets.append(offset)
        search_start = offset.end
    return offsets


def validate_label_mapping(
    label_to_id: Mapping[str, int], *, ignore_index: int = IGNORE_INDEX
) -> None:
    """Validate a unique BIO-label to model-class mapping."""

    if not isinstance(label_to_id, Mapping):
        raise TypeError("label_to_id must be a mapping")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")

    ids: list[int] = []
    for label, label_id in label_to_id.items():
        parse_bio_label(label)
        if isinstance(label_id, bool) or not isinstance(label_id, int):
            raise TypeError(f"ID for label {label!r} must be an integer")
        if label_id == ignore_index:
            raise ValueError(f"label ID for {label!r} must not equal ignore_index")
        ids.append(label_id)
    if len(ids) != len(set(ids)):
        raise ValueError("label IDs must be unique")


def labels_to_ids(
    token_labels: Sequence[TokenLabel],
    label_to_id: Mapping[str, int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[int, ...]:
    """Convert token labels to numeric IDs, mapping ``None`` to ignore index."""

    validate_label_mapping(label_to_id, ignore_index=ignore_index)
    converted: list[int] = []
    for index, label in enumerate(token_labels):
        if label is None:
            converted.append(ignore_index)
            continue
        parse_bio_label(label)
        if label not in label_to_id:
            raise ValueError(f"label {label!r} at token index {index} is missing from label_to_id")
        converted.append(label_to_id[label])
    return tuple(converted)


def align_bio_to_subwords(
    text: str,
    words: Sequence[str],
    labels: Sequence[str],
    tokenizer: FastTokenizer,
    strategy: SubwordLabelStrategy = "all_subwords",
    max_length: int | None = None,
) -> AlignedExample:
    """Project word BIO labels and local subword offsets onto source text.

    ``all_subwords`` labels continuation pieces with ``I-<TYPE>``;
    ``first_subword`` marks them ``None`` so a loss function can ignore them.
    Overlong examples fail explicitly instead of being silently truncated.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not callable(tokenizer) or not hasattr(tokenizer, "unk_token"):
        raise TypeError("tokenizer must provide a fast-tokenizer compatible interface")
    if len(words) != len(labels):
        raise ValueError("words and labels must contain the same number of items")
    if strategy not in {"all_subwords", "first_subword"}:
        raise ValueError("strategy must be 'all_subwords' or 'first_subword'")
    if max_length is not None:
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer or None")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")

    word_offsets = locate_words(text, words)
    parsed = [parse_bio_label(label) for label in labels]
    encoding = tokenizer(
        list(words),
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = list(encoding["input_ids"])
    attention_mask = list(encoding["attention_mask"])
    local_offsets = list(encoding["offset_mapping"])
    word_ids = encoding.word_ids()
    tokens = encoding.tokens()
    if word_ids is None:
        raise ValueError("fast tokenizer did not return word IDs")
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(
            f"tokenized example contains {len(input_ids)} tokens, which exceeds "
            f"max_length={max_length}; window the input instead of truncating it"
        )
    if len(word_ids) != len(local_offsets):
        raise ValueError("tokenizer word IDs and offsets have different lengths")
    if len(tokens) != len(word_ids):
        raise ValueError("tokenizer tokens and word IDs have different lengths")

    global_offsets: list[tuple[int, int]] = []
    token_labels: list[str | None] = []
    seen_first: set[int] = set()
    coverage: dict[int, list[tuple[int, int]]] = {}
    for token_index, (word_id, local_offset) in enumerate(
        zip(word_ids, local_offsets, strict=True)
    ):
        if word_id is None:
            global_offsets.append((0, 0))
            token_labels.append(None)
            continue
        if isinstance(word_id, bool) or not isinstance(word_id, int):
            raise TypeError(f"word ID at token index {token_index} must be an integer or None")
        if word_id < 0 or word_id >= len(words):
            raise ValueError(f"token refers to invalid word index {word_id}")
        local_start, local_end = local_offset
        if local_start < 0 or local_end < local_start:
            raise ValueError(f"token offset {local_offset} at index {token_index} is invalid")

        prefix, entity_type = parsed[word_id]
        if (
            tokenizer.unk_token is not None
            and tokens[token_index] == tokenizer.unk_token
            and prefix != "O"
        ):
            raise ValueError(
                f"annotated word {words[word_id]!r} at index {word_id} "
                "was converted to the unknown token"
            )

        source = word_offsets[word_id]
        global_start, global_end = source.start + local_start, source.start + local_end
        if global_start < source.start:
            raise ValueError(f"token offset at index {token_index} starts before its source word")
        if global_end > source.end:
            raise ValueError(f"token offset at index {token_index} ends after its source word")
        global_offsets.append((global_start, global_end))
        coverage.setdefault(word_id, []).append((local_start, local_end))

        if word_id not in seen_first:
            seen_first.add(word_id)
            token_labels.append(labels[word_id])
        elif strategy == "first_subword":
            token_labels.append(None)
        else:
            token_labels.append("O" if prefix == "O" else f"I-{entity_type}")

    missing = set(range(len(words))) - set(coverage)
    if missing:
        raise ValueError(
            f"tokenizer dropped source words: {[words[index] for index in sorted(missing)]}"
        )
    _validate_coverage(words, coverage)
    return AlignedExample(
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        offset_mapping=tuple(global_offsets),
        word_ids=tuple(word_ids),
        token_labels=tuple(token_labels),
    )


def _validate_coverage(words: Sequence[str], coverage: dict[int, list[tuple[int, int]]]) -> None:
    """Reject token offsets that omit, overlap, or extend source characters."""

    for word_id, word in enumerate(words):
        ranges = sorted(coverage.get(word_id, []))
        if not ranges:
            raise ValueError(f"tokenizer produced no offsets for word {word!r} at index {word_id}")
        if ranges[0][0] != 0 or max(end for _, end in ranges) != len(word):
            raise ValueError(f"token offsets do not fully cover word {word!r} at index {word_id}")

        covered_end = ranges[0][1]
        for start, end in ranges[1:]:
            if start > covered_end:
                raise ValueError(
                    f"token offsets contain a gap for word {word!r} at index {word_id}"
                )
            covered_end = max(covered_end, end)
