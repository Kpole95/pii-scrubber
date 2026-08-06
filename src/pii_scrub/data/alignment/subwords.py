"""Align word-level BIO labels with tokenizer subwords."""

from collections.abc import Sequence

from transformers import PreTrainedTokenizerFast

from .constants import ALL_SUBWORDS, FIRST_SUBWORD
from .models import AlignedExample
from .types import SubwordLabelStrategy
from .validation import parse_bio_label
from .words import locate_words


def _validate_strategy(strategy: str) -> None:
    """Reject unsupported subword-label strategies."""

    if strategy not in {ALL_SUBWORDS, FIRST_SUBWORD}:
        raise ValueError(
            "strategy must be 'all_subwords' or 'first_subword'"
        )


def _validate_max_length(max_length: int | None) -> None:
    """Validate an optional maximum token count."""

    if max_length is None:
        return

    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer or None")

    if max_length <= 0:
        raise ValueError("max_length must be greater than zero")


def _validate_word_coverage(
    *,
    words: Sequence[str],
    covered_ranges_by_word: dict[int, list[tuple[int, int]]],
) -> None:
    """Ensure tokenizer offsets cover every character of every word."""

    for word_id, word in enumerate(words):
        ranges = sorted(covered_ranges_by_word.get(word_id, []))

        if not ranges:
            raise ValueError(
                f"tokenizer produced no offsets for word "
                f"{word!r} at index {word_id}"
            )

        if ranges[0][0] != 0 or ranges[-1][1] != len(word):
            raise ValueError(
                f"token offsets do not fully cover word "
                f"{word!r} at index {word_id}"
            )

        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] != current[0]:
                raise ValueError(
                    f"token offsets contain a gap or overlap for word "
                    f"{word!r} at index {word_id}"
                )


def align_bio_to_subwords(
    text: str,
    words: Sequence[str],
    labels: Sequence[str],
    tokenizer: PreTrainedTokenizerFast,
    strategy: SubwordLabelStrategy = ALL_SUBWORDS,
    max_length: int | None = None,
) -> AlignedExample:
    """Align word-level BIO labels with fast-tokenizer subwords.

    Two strategies are supported:

    - ``all_subwords`` labels every subword.
    - ``first_subword`` labels only the first piece of each word.

    Example:
        ``Murali/B-PERSON`` may become:

        - ``Mur/B-PERSON``
        - ``##ali/I-PERSON``
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            "tokenizer must be a PreTrainedTokenizerFast"
        )

    if len(words) != len(labels):
        raise ValueError(
            "words and labels must contain the same number of items"
        )

    _validate_strategy(strategy)
    _validate_max_length(max_length)

    word_offsets = locate_words(text, words)
    parsed_labels = [parse_bio_label(label) for label in labels]

    encoding = tokenizer(
        list(words),
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

    if len(word_ids) != len(local_offsets):
        raise ValueError(
            "tokenizer word IDs and offsets have different lengths"
        )

    if len(tokens) != len(word_ids):
        raise ValueError(
            "tokenizer tokens and word IDs have different lengths"
        )

    global_offsets: list[tuple[int, int]] = []
    token_labels: list[str | None] = []
    seen_word_ids: set[int] = set()
    seen_first_subword: set[int] = set()
    covered_ranges_by_word: dict[
        int,
        list[tuple[int, int]],
    ] = {}

    for token_index, (word_id, local_offset) in enumerate(
        zip(word_ids, local_offsets, strict=True)
    ):
        if word_id is None:
            global_offsets.append((0, 0))
            token_labels.append(None)
            continue

        if isinstance(word_id, bool) or not isinstance(word_id, int):
            raise TypeError(
                f"word ID at token index {token_index} "
                "must be an integer or None"
            )

        if word_id < 0 or word_id >= len(words):
            raise ValueError(
                f"token refers to invalid word index {word_id}"
            )

        local_start, local_end = local_offset

        if local_start < 0 or local_end < local_start:
            raise ValueError(
                f"token offset {local_offset} at index "
                f"{token_index} is invalid"
            )

        token_text = tokens[token_index]
        prefix, entity_type = parsed_labels[word_id]

        if (
            tokenizer.unk_token is not None
            and token_text == tokenizer.unk_token
            and prefix != "O"
        ):
            raise ValueError(
                f"annotated word {words[word_id]!r} "
                f"at index {word_id} was converted "
                "to the unknown token"
            )

        word_offset = word_offsets[word_id]
        global_start = word_offset.start + local_start
        global_end = word_offset.start + local_end

        if global_start < word_offset.start:
            raise ValueError(
                f"token offset at index {token_index} "
                "starts before its source word"
            )

        if global_end > word_offset.end:
            raise ValueError(
                f"token offset at index {token_index} "
                "ends after its source word"
            )

        global_offsets.append((global_start, global_end))
        seen_word_ids.add(word_id)

        covered_ranges_by_word.setdefault(word_id, []).append(
            (local_start, local_end)
        )

        is_first_subword = word_id not in seen_first_subword

        if is_first_subword:
            seen_first_subword.add(word_id)
            token_labels.append(labels[word_id])
            continue

        if strategy == FIRST_SUBWORD:
            token_labels.append(None)
            continue

        if prefix == "O":
            token_labels.append("O")
        else:
            token_labels.append(f"I-{entity_type}")

    missing_word_ids = set(range(len(words))) - seen_word_ids

    if missing_word_ids:
        missing_words = [
            words[index]
            for index in sorted(missing_word_ids)
        ]

        raise ValueError(
            f"tokenizer dropped source words: {missing_words}"
        )

    _validate_word_coverage(
        words=words,
        covered_ranges_by_word=covered_ranges_by_word,
    )

    return AlignedExample(
        input_ids=tuple(encoding["input_ids"]),
        attention_mask=tuple(encoding["attention_mask"]),
        offset_mapping=tuple(global_offsets),
        word_ids=tuple(word_ids),
        token_labels=tuple(token_labels),
    )