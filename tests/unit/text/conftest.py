"""Deterministic tokenizer fixtures with no network or ML dependency."""

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class FakeEncoding:
    """Mimic the small fast-tokenizer encoding surface used by alignment."""

    data: dict[str, list[Any]]
    _word_ids: list[int | None]
    _tokens: list[str]

    def __getitem__(self, key: str) -> list[Any]:
        return self.data[key]

    def word_ids(self) -> list[int | None]:
        return self._word_ids

    def tokens(self) -> list[str]:
        return self._tokens


class FakeFastTokenizer:
    """Tokenize pre-split words with a minimal greedy WordPiece algorithm."""

    unk_token: str | None = "[UNK]"

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary = vocabulary

    def __call__(self, words: list[str], **_: Any) -> FakeEncoding:
        tokens = ["[CLS]"]
        word_ids: list[int | None] = [None]
        offsets = [(0, 0)]
        for word_id, word in enumerate(words):
            pieces = self._split(word)
            cursor = 0
            for piece in pieces:
                surface = piece[2:] if piece.startswith("##") else piece
                end = len(word) if piece == self.unk_token else cursor + len(surface)
                tokens.append(piece)
                word_ids.append(word_id)
                offsets.append((cursor, end))
                cursor = end
        tokens.append("[SEP]")
        word_ids.append(None)
        offsets.append((0, 0))
        return FakeEncoding(
            data={
                "input_ids": [self.vocabulary[token] for token in tokens],
                "attention_mask": [1] * len(tokens),
                "offset_mapping": offsets,
            },
            _word_ids=word_ids,
            _tokens=tokens,
        )

    def _split(self, word: str) -> list[str]:
        if word in self.vocabulary:
            return [word]
        pieces: list[str] = []
        cursor = 0
        while cursor < len(word):
            match = None
            for end in range(len(word), cursor, -1):
                candidate = word[cursor:end]
                token = candidate if cursor == 0 else f"##{candidate}"
                if token in self.vocabulary:
                    match = token
                    break
            if match is None:
                assert self.unk_token is not None
                return [self.unk_token]
            pieces.append(match)
            cursor = end
        return pieces


def _build_tokenizer(vocabulary: dict[str, int]) -> FakeFastTokenizer:
    return FakeFastTokenizer(vocabulary)


@pytest.fixture
def toy_tokenizer() -> FakeFastTokenizer:
    """Create a tokenizer that splits ``Murali`` into ``Mur`` and ``##ali``."""

    return _build_tokenizer(
        {
            "[UNK]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[PAD]": 3,
            "Call": 4,
            "Mur": 5,
            "##ali": 6,
            "Krishna": 7,
            "today": 8,
            ".": 9,
            "met": 10,
        }
    )


@pytest.fixture
def unicode_toy_tokenizer() -> FakeFastTokenizer:
    return _build_tokenizer(
        {
            "[UNK]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[PAD]": 3,
            "Contact": 4,
            "José": 5,
            "Álvarez": 6,
            "today": 7,
            ".": 8,
        }
    )


@pytest.fixture
def limited_toy_tokenizer() -> FakeFastTokenizer:
    return _build_tokenizer(
        {
            "[UNK]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[PAD]": 3,
            "Call": 4,
            "today": 5,
            ".": 6,
        }
    )
