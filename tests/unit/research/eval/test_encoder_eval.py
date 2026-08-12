"""Tests for encoder evaluation inference helpers."""

import pytest

from pii_scrub.text import merge_window_predictions, token_predictions_to_spans
from pii_scrub.types import CharacterSpan, DetectedSpan, WindowSpan
from research.eval.encoder import _project_offsets, _split_words


def test_prediction_pipeline_decodes_and_merges_duplicate_windows() -> None:
    """Overlapping windows should produce one best document span."""

    first = token_predictions_to_spans(
        offsets=[(10, 14), (14, 18)],
        labels=["B-PERSON", "I-PERSON"],
        scores=[0.8, 0.8],
    )
    second = token_predictions_to_spans(
        offsets=[(10, 14), (14, 18)],
        labels=["B-PERSON", "I-PERSON"],
        scores=[0.9, 0.9],
    )

    predictions = [
        WindowSpan(0, CharacterSpan(span.start, span.end, span.entity_type), span.score)
        for span in first
    ]
    predictions.extend(
        WindowSpan(1, CharacterSpan(span.start, span.end, span.entity_type), span.score)
        for span in second
    )

    assert merge_window_predictions(predictions) == [
        WindowSpan(1, CharacterSpan(10, 18, "PERSON"), 0.9)
    ]


def test_prediction_pipeline_supports_entity_filter() -> None:
    """Encoder output should be filterable using detector entity names."""

    predictions = [
        DetectedSpan(0, 4, "PERSON", 0.9),
        DetectedSpan(10, 20, "EMAIL", 0.8),
    ]

    assert [span for span in predictions if span.entity_type in {"EMAIL"}] == [
        DetectedSpan(10, 20, "EMAIL", 0.8)
    ]


def test_split_words_preserves_source_offsets() -> None:
    """Inference words should retain their exact positions in source text."""

    words, offsets = _split_words("Call  John Smith.")

    assert words == ("Call", "John", "Smith.")
    assert offsets == ((0, 4), (6, 10), (11, 17))


def test_project_offsets_removes_sentencepiece_whitespace_shift() -> None:
    """Word-local offsets should map onto source words, not preceding spaces."""

    offsets = _project_offsets(
        [(0, 0), (0, 4), (0, 5), (0, 0)],
        [None, 0, 1, None],
        [(8, 12), (13, 18)],
    )

    assert offsets == [(0, 0), (8, 12), (13, 18), (0, 0)]


def test_project_offsets_preserves_overlapping_sentencepiece_pieces() -> None:
    """Overlapping local pieces should remain valid after global projection."""

    assert _project_offsets(
        [(0, 1), (0, 1), (1, 9)],
        [0, 0, 0],
        [(12, 21)],
    ) == [
        (12, 13),
        (12, 13),
        (13, 21),
    ]


def test_project_offsets_rejects_invalid_word_id() -> None:
    """Tokenizer word IDs must refer to real source words."""

    with pytest.raises(ValueError, match="invalid word index"):
        _project_offsets([(0, 2)], [2], [(0, 4)])


def test_project_offsets_rejects_out_of_word_range() -> None:
    """Subword offsets must stay inside their corresponding source word."""

    with pytest.raises(ValueError, match="exceeds its source word"):
        _project_offsets([(0, 5)], [0], [(10, 14)])
