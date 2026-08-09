"""Prepare normalized examples for encoder token classification."""

import re
from collections.abc import Iterable, Mapping
from itertools import pairwise

from pii_scrub.text.alignment import FastTokenizer, align_bio_to_subwords, labels_to_ids
from pii_scrub.types import AlignedExample, CharacterSpan, SubwordLabelStrategy
from research.data.models import DatasetExample

_NON_SPACE = re.compile(r"\S+")


def words_and_bio(example: DatasetExample) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Convert character spans into word-level BIO labels."""
    boundaries = {x for span in example.spans for x in (span.start, span.end)}
    pieces: list[tuple[str, int, int]] = []

    for match in _NON_SPACE.finditer(example.text):
        cuts = sorted(
            {match.start(), match.end()}
            | {x for x in boundaries if match.start() < x < match.end()}
        )
        pieces.extend((example.text[start:end], start, end) for start, end in pairwise(cuts))

    words = tuple(word for word, _, _ in pieces)
    labels = tuple(_bio_label(start, end, example.spans) for _, start, end in pieces)
    return words, labels


def window_example(
    example: DatasetExample,
    tokenizer: FastTokenizer,
    *,
    max_length: int,
    overlap: int,
) -> list[DatasetExample]:
    """Split one long example into tokenizer-safe windows."""
    words, labels = words_and_bio(example)
    offsets = _word_offsets(example, words)
    windows: list[DatasetExample] = []
    start = 0

    while start < len(words):
        end = _largest_end(words, tokenizer, start, max_length)
        while end < len(words) and labels[end].startswith("I-"):
            end -= 1
        if end <= start:
            raise ValueError(f"entity exceeds max_length={max_length}")

        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        spans = tuple(
            CharacterSpan(span.start - char_start, span.end - char_start, span.entity_type)
            for span in example.spans
            if char_start <= span.start and span.end <= char_end
        )
        windows.append(
            DatasetExample(
                f"{example.example_id}-w{len(windows)}",
                example.text[char_start:char_end],
                spans,
                example.source,
                example.language,
            )
        )

        if end == len(words):
            break

        start = _overlap_start(words, tokenizer, start, end, overlap)
        while start > 0 and labels[start].startswith("I-"):
            start -= 1

    return windows


def build_label_mapping(examples: Iterable[DatasetExample]) -> dict[str, int]:
    """Build deterministic BIO label IDs."""
    entities = sorted({span.entity_type for example in examples for span in example.spans})
    labels = ["O"] + [label for entity in entities for label in (f"B-{entity}", f"I-{entity}")]
    return {label: index for index, label in enumerate(labels)}


def prepare_encoder_example(
    example: DatasetExample,
    tokenizer: FastTokenizer,
    label_to_id: Mapping[str, int],
    *,
    strategy: SubwordLabelStrategy,
    max_length: int,
) -> tuple[AlignedExample, tuple[int, ...]]:
    """Create aligned encoder inputs and label IDs."""
    words, labels = words_and_bio(example)
    aligned = align_bio_to_subwords(
        example.text, words, labels, tokenizer, strategy=strategy, max_length=max_length
    )
    return aligned, labels_to_ids(aligned.token_labels, label_to_id)


def _word_offsets(
    example: DatasetExample,
    words: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    cursor = 0
    offsets = []

    for word in words:
        start = example.text.index(word, cursor)
        end = start + len(word)
        offsets.append((start, end))
        cursor = end

    return tuple(offsets)


def _token_count(words: tuple[str, ...], tokenizer: FastTokenizer) -> int:
    encoding = tokenizer(
        list(words),
        is_split_into_words=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    return len(encoding["input_ids"])


def _largest_end(
    words: tuple[str, ...],
    tokenizer: FastTokenizer,
    start: int,
    max_length: int,
) -> int:
    end = start + 1

    while end <= len(words) and _token_count(words[start:end], tokenizer) <= max_length:
        end += 1

    return min(end - 1, len(words))


def _overlap_start(
    words: tuple[str, ...],
    tokenizer: FastTokenizer,
    start: int,
    end: int,
    overlap: int,
) -> int:
    next_start = end - 1

    while next_start > start and _token_count(words[next_start:end], tokenizer) < overlap:
        next_start -= 1

    return next_start


def _bio_label(start: int, end: int, spans: tuple[CharacterSpan, ...]) -> str:
    for span in spans:
        if span.start <= start and end <= span.end:
            prefix = "B" if start == span.start else "I"
            return f"{prefix}-{span.entity_type}"
    return "O"


def prepare_encoder_records(
    examples: Iterable[DatasetExample],
    tokenizer: FastTokenizer,
    label_to_id: Mapping[str, int],
    *,
    strategy: SubwordLabelStrategy,
    max_length: int,
    overlap: int,
) -> list[dict[str, list[int]]]:
    """Convert examples into model-ready token-classification records."""
    records = []

    for example in examples:
        for window in window_example(example, tokenizer, max_length=max_length, overlap=overlap):
            aligned, labels = prepare_encoder_example(
                window,
                tokenizer,
                label_to_id,
                strategy=strategy,
                max_length=max_length,
            )
            records.append(
                {
                    "input_ids": list(aligned.input_ids),
                    "attention_mask": list(aligned.attention_mask),
                    "labels": list(labels),
                }
            )

    return records
