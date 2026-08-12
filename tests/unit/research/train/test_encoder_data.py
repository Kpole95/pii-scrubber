"""Tests for encoder training-data preparation."""

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.train.encoder_data import (
    build_label_mapping,
    prepare_encoder_records,
    window_example,
    words_and_bio,
)
from tests.unit.text.conftest import FakeFastTokenizer


def _tokenizer() -> FakeFastTokenizer:
    """Build the tokenizer fake used by encoder-data tests."""
    return FakeFastTokenizer(
        {
            "[UNK]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[PAD]": 3,
            "Call": 4,
            "Mur": 5,
            "##ali": 6,
            "today": 7,
            ".": 8,
        }
    )


def test_builds_bio_for_multiword_entity() -> None:
    """Check BIO labels for a multiword entity."""
    example = DatasetExample(
        "one", "Call John Smith today.", (CharacterSpan(5, 15, "PERSON"),), "test", "en"
    )
    assert words_and_bio(example) == (
        ("Call", "John", "Smith", "today."),
        ("O", "B-PERSON", "I-PERSON", "O"),
    )


def test_splits_at_entity_boundary() -> None:
    """Check that encoder windows split at entity boundaries."""
    example = DatasetExample(
        "one", "Email ana@example.com.", (CharacterSpan(6, 21, "EMAIL"),), "test", "en"
    )
    assert words_and_bio(example) == (
        ("Email", "ana@example.com", "."),
        ("O", "B-EMAIL", "O"),
    )


def test_handles_entity_inside_text_piece() -> None:
    """Check an entity that appears inside a text piece."""
    example = DatasetExample("one", "User:John!", (CharacterSpan(5, 9, "PERSON"),), "test", "en")
    assert words_and_bio(example) == (
        ("User:", "John", "!"),
        ("O", "B-PERSON", "O"),
    )


def test_builds_deterministic_label_mapping() -> None:
    """Check that label mapping is deterministic."""
    examples = (
        DatasetExample("one", "a@b.com", (CharacterSpan(0, 7, "EMAIL"),), "test", "en"),
        DatasetExample("two", "John", (CharacterSpan(0, 4, "PERSON"),), "test", "en"),
    )
    assert build_label_mapping(examples) == {
        "O": 0,
        "B-EMAIL": 1,
        "I-EMAIL": 2,
        "B-PERSON": 3,
        "I-PERSON": 4,
    }


def test_window_example_keeps_short_example_whole() -> None:
    """Check that short examples stay in one window."""
    example = DatasetExample(
        "one", "Call Murali today.", (CharacterSpan(5, 11, "PERSON"),), "test", "en"
    )
    windows = window_example(example, _tokenizer(), max_length=10, overlap=2)

    assert len(windows) == 1
    assert windows[0].text == example.text
    assert windows[0].spans == example.spans


def test_window_example_rebases_spans() -> None:
    """Check that windowed spans are rebased correctly."""
    example = DatasetExample(
        "one",
        "Call Murali today. Call Murali today.",
        (CharacterSpan(24, 30, "PERSON"),),
        "test",
        "en",
    )
    windows = window_example(example, _tokenizer(), max_length=7, overlap=2)

    assert any(span.extract(window.text) == "Murali" for window in windows for span in window.spans)


def test_window_example_respects_max_length() -> None:
    """Check that windows respect the maximum token length."""
    example = DatasetExample("one", "Call Murali today. Call Murali today.", (), "test", "en")
    tokenizer = _tokenizer()
    windows = window_example(example, tokenizer, max_length=7, overlap=2)

    for window in windows:
        words, _ = words_and_bio(window)
        assert len(tokenizer(list(words))["input_ids"]) <= 7


def test_prepare_encoder_records_builds_model_inputs() -> None:
    """Check that encoder records contain the required model inputs."""
    example = DatasetExample(
        "one", "Call Murali today.", (CharacterSpan(5, 11, "PERSON"),), "test", "en"
    )
    labels = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}

    records = prepare_encoder_records(
        (example,),
        _tokenizer(),
        labels,
        strategy="all_subwords",
        max_length=10,
        overlap=2,
    )

    assert set(records[0]) == {"input_ids", "attention_mask", "labels"}
    assert len(records[0]["input_ids"]) == len(records[0]["labels"])


def test_prepare_encoder_records_windows_long_examples() -> None:
    """Check that long encoder examples are windowed."""
    example = DatasetExample("one", "Call Murali today. Call Murali today.", (), "test", "en")

    records = prepare_encoder_records(
        (example,),
        _tokenizer(),
        {"O": 0},
        strategy="all_subwords",
        max_length=7,
        overlap=2,
    )

    assert len(records) > 1
    assert all(len(record["input_ids"]) <= 7 for record in records)
