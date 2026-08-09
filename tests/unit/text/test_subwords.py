"""Tests for word BIO to tokenizer-subword alignment."""

import pytest

from pii_scrub.text import align_bio_to_subwords
from tests.unit.text.conftest import FakeEncoding, FakeFastTokenizer


def test_all_subwords_propagates_labels(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Later entity subwords should receive I labels."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.token_labels == (
        None,
        "O",
        "B-PERSON",
        "I-PERSON",
        "I-PERSON",
        "O",
        "O",
        None,
    )


def test_first_subword_ignores_later_pieces(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Only the first piece of each source word should be labeled."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
        strategy="first_subword",
    )

    assert example.token_labels == (
        None,
        "O",
        "B-PERSON",
        None,
        "I-PERSON",
        "O",
        "O",
        None,
    )


def test_alignment_returns_global_text_offsets(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Token offsets must refer to the original full text."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.offset_mapping == (
        (0, 0),
        (0, 4),
        (5, 8),
        (8, 11),
        (12, 19),
        (20, 25),
        (25, 26),
        (0, 0),
    )


def test_alignment_preserves_word_ids(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Subwords should remember their original source-word index."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.word_ids == (
        None,
        0,
        1,
        1,
        2,
        3,
        4,
        None,
    )


def test_special_tokens_have_no_labels(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """CLS and SEP should not receive training labels."""

    example = align_bio_to_subwords(
        text="Call Murali.",
        words=["Call", "Murali", "."],
        labels=["O", "B-PERSON", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.word_ids[0] is None
    assert example.offset_mapping[0] == (0, 0)
    assert example.token_labels[0] is None

    assert example.word_ids[-1] is None
    assert example.offset_mapping[-1] == (0, 0)
    assert example.token_labels[-1] is None


def test_alignment_rejects_unknown_strategy(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Only the two documented strategies are supported."""

    with pytest.raises(
        ValueError,
        match="strategy must be 'all_subwords' or 'first_subword'",
    ):
        align_bio_to_subwords(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "B-PERSON", "O"],
            tokenizer=toy_tokenizer,
            strategy="unknown",  # type: ignore[arg-type]
        )


def test_alignment_rejects_unknown_pii_word(
    limited_toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Annotated PII must not silently become an unknown token."""

    with pytest.raises(
        ValueError,
        match=r"annotated word 'Murali'.*unknown token",
    ):
        align_bio_to_subwords(
            text="Call Murali today.",
            words=["Call", "Murali", "today", "."],
            labels=["O", "B-PERSON", "O", "O"],
            tokenizer=limited_toy_tokenizer,
        )


def test_alignment_allows_unknown_non_pii_word(
    limited_toy_tokenizer: FakeFastTokenizer,
) -> None:
    """An O-labeled unknown word may be retained."""

    example = align_bio_to_subwords(
        text="Call unusualword today.",
        words=["Call", "unusualword", "today", "."],
        labels=["O", "O", "O", "O"],
        tokenizer=limited_toy_tokenizer,
    )

    assert example.token_count == 6


def test_split_word_offsets_cover_every_character(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """Mur and ##ali should cover all six characters of Murali."""

    example = align_bio_to_subwords(
        text="Murali",
        words=["Murali"],
        labels=["B-PERSON"],
        tokenizer=toy_tokenizer,
    )

    offsets = [
        offset
        for offset, word_id in zip(
            example.offset_mapping,
            example.word_ids,
            strict=True,
        )
        if word_id == 0
    ]

    assert offsets == [(0, 3), (3, 6)]


def test_alignment_accepts_input_within_max_length(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """A six-token example should fit max_length=6."""

    example = align_bio_to_subwords(
        text="Call Murali.",
        words=["Call", "Murali", "."],
        labels=["O", "B-PERSON", "O"],
        tokenizer=toy_tokenizer,
        max_length=6,
    )

    assert example.token_count == 6


def test_alignment_rejects_overlong_input(
    toy_tokenizer: FakeFastTokenizer,
) -> None:
    """An overlong sequence must fail instead of being truncated."""

    with pytest.raises(
        ValueError,
        match=r"contains 6 tokens.*max_length=5",
    ):
        align_bio_to_subwords(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "B-PERSON", "O"],
            tokenizer=toy_tokenizer,
            max_length=5,
        )


@pytest.mark.parametrize("max_length", [0, -1])
def test_alignment_rejects_non_positive_max_length(
    toy_tokenizer: FakeFastTokenizer,
    max_length: int,
) -> None:
    """Maximum length must be greater than zero."""

    with pytest.raises(
        ValueError,
        match="max_length must be greater than zero",
    ):
        align_bio_to_subwords(
            text="Call.",
            words=["Call", "."],
            labels=["O", "O"],
            tokenizer=toy_tokenizer,
            max_length=max_length,
        )


@pytest.mark.parametrize("max_length", [5.5, "5", True])
def test_alignment_rejects_invalid_max_length_type(
    toy_tokenizer: FakeFastTokenizer,
    max_length: object,
) -> None:
    """Maximum length must be an integer or None."""

    with pytest.raises(
        TypeError,
        match="max_length must be an integer or None",
    ):
        align_bio_to_subwords(
            text="Call.",
            words=["Call", "."],
            labels=["O", "O"],
            tokenizer=toy_tokenizer,
            max_length=max_length,  # type: ignore[arg-type]
        )


def test_alignment_allows_overlapping_offsets_without_gaps() -> None:
    """SentencePiece-style overlap is valid when all characters are covered."""

    class OverlapTokenizer(FakeFastTokenizer):
        def __call__(self, words: list[str], **kwargs: object) -> FakeEncoding:
            encoding = super().__call__(words, **kwargs)
            encoding.data["offset_mapping"] = [(0, 0), (0, 1), (0, 1), (1, 9), (0, 0)]
            encoding._word_ids = [None, 0, 0, 0, None]
            encoding._tokens = ["[CLS]", "▁", "₨", "Pakistan", "[SEP]"]
            encoding.data["input_ids"] = [1, 4, 5, 6, 2]
            encoding.data["attention_mask"] = [1] * 5
            return encoding

    tokenizer = OverlapTokenizer(
        {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "▁": 4, "₨": 5, "Pakistan": 6}
    )
    example = align_bio_to_subwords("₨Pakistan", ["₨Pakistan"], ["B-LOCATION"], tokenizer)

    assert example.token_labels == (None, "B-LOCATION", "I-LOCATION", "I-LOCATION", None)
