"""End-to-end round-trip tests for the alignment package."""

import pytest

from pii_scrub.text import (
    align_bio_to_subwords,
    aligned_labels_to_spans,
    bio_tags_to_spans,
)
from pii_scrub.types import SubwordLabelStrategy
from tests.unit.text.conftest import FakeFastTokenizer


def _assert_round_trip(
    *,
    text: str,
    words: list[str],
    labels: list[str],
    tokenizer: FakeFastTokenizer,
    strategy: SubwordLabelStrategy,
) -> None:
    """Verify that alignment preserves the original entity spans.

    Example:
        ``Murali Krishna`` starts as one PERSON span.
        After tokenization and reconstruction, it must remain the same span.
    """

    gold_spans = bio_tags_to_spans(
        text=text,
        words=words,
        labels=labels,
    )

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=tokenizer,
        strategy=strategy,
    )

    reconstructed_spans = aligned_labels_to_spans(aligned)

    assert reconstructed_spans == gold_spans

    assert [span.extract(text) for span in reconstructed_spans] == [
        span.extract(text) for span in gold_spans
    ]


@pytest.mark.parametrize(
    "strategy",
    [
        "all_subwords",
        "first_subword",
    ],
)
def test_multiword_entity_round_trips(
    toy_tokenizer: FakeFastTokenizer,
    strategy: SubwordLabelStrategy,
) -> None:
    """Both strategies must preserve one multiword entity."""

    _assert_round_trip(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
        strategy=strategy,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "all_subwords",
        "first_subword",
    ],
)
def test_split_single_word_entity_round_trips(
    toy_tokenizer: FakeFastTokenizer,
    strategy: SubwordLabelStrategy,
) -> None:
    """A split word must keep its complete character boundary."""

    _assert_round_trip(
        text="Call Murali today.",
        words=["Call", "Murali", "today", "."],
        labels=["O", "B-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
        strategy=strategy,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "all_subwords",
        "first_subword",
    ],
)
def test_multiple_entities_round_trip(
    toy_tokenizer: FakeFastTokenizer,
    strategy: SubwordLabelStrategy,
) -> None:
    """Separate entities must remain separate."""

    _assert_round_trip(
        text="Murali met Krishna.",
        words=["Murali", "met", "Krishna", "."],
        labels=["B-PERSON", "O", "B-PERSON", "O"],
        tokenizer=toy_tokenizer,
        strategy=strategy,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "all_subwords",
        "first_subword",
    ],
)
def test_irregular_whitespace_round_trips(
    toy_tokenizer: FakeFastTokenizer,
    strategy: SubwordLabelStrategy,
) -> None:
    """Tabs, newlines, and repeated spaces must preserve offsets."""

    _assert_round_trip(
        text="Call\tMurali\nKrishna  today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
        strategy=strategy,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "all_subwords",
        "first_subword",
    ],
)
def test_unicode_entity_round_trips(
    unicode_toy_tokenizer: FakeFastTokenizer,
    strategy: SubwordLabelStrategy,
) -> None:
    """Unicode entity offsets must survive tokenization."""

    _assert_round_trip(
        text="Contact José Álvarez today.",
        words=["Contact", "José", "Álvarez", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=unicode_toy_tokenizer,
        strategy=strategy,
    )
