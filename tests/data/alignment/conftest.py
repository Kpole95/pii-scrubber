"""Shared deterministic tokenizer fixtures."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def _build_tokenizer(
    vocabulary: dict[str, int],
) -> PreTrainedTokenizerFast:
    """Build a local WordPiece tokenizer without network access."""

    backend = Tokenizer(
        WordPiece(
            vocab=vocabulary,
            unk_token="[UNK]",
        )
    )

    backend.pre_tokenizer = BertPreTokenizer()

    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", vocabulary["[CLS]"]),
            ("[SEP]", vocabulary["[SEP]"]),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )


@pytest.fixture
def toy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a tokenizer that splits ``Murali`` into two pieces."""

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
def unicode_toy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a tokenizer containing Unicode name tokens."""

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
def limited_toy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a tokenizer that cannot represent names such as Murali."""

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