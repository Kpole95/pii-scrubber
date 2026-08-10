"""Tests for converting model BIO predictions into detected spans."""

import pytest

from pii_scrub.text import token_predictions_to_spans
from pii_scrub.types import DetectedSpan


def test_token_predictions_build_scored_entity() -> None:
    """Subword predictions should become one scored character span."""

    assert token_predictions_to_spans(
        offsets=[(0, 0), (5, 8), (8, 11), (0, 0)],
        labels=["O", "B-PERSON", "I-PERSON", "O"],
        scores=[0.99, 0.8, 0.6, 0.99],
    ) == [DetectedSpan(5, 11, "PERSON", 0.7)]


def test_token_predictions_support_overlapping_sentencepiece_offsets() -> None:
    """Overlapping pieces should preserve the union of source characters."""

    spans = token_predictions_to_spans(
        offsets=[(0, 1), (0, 1), (1, 9)],
        labels=["B-LOCATION", "I-LOCATION", "I-LOCATION"],
        scores=[0.9, 0.8, 0.7],
    )

    assert spans[0].start == 0
    assert spans[0].end == 9
    assert spans[0].entity_type == "LOCATION"
    assert spans[0].score == pytest.approx(0.8)


def test_token_predictions_repair_orphan_i() -> None:
    """An orphan I prediction should start an entity instead of crashing."""

    assert token_predictions_to_spans(
        offsets=[(5, 11)],
        labels=["I-PERSON"],
        scores=[0.75],
    ) == [DetectedSpan(5, 11, "PERSON", 0.75)]


def test_token_predictions_repair_i_type_change() -> None:
    """A changed I type should close the old entity and start the new one."""

    assert token_predictions_to_spans(
        offsets=[(0, 6), (7, 13)],
        labels=["B-PERSON", "I-LOCATION"],
        scores=[0.9, 0.8],
    ) == [
        DetectedSpan(0, 6, "PERSON", 0.9),
        DetectedSpan(7, 13, "LOCATION", 0.8),
    ]


def test_token_predictions_ignore_zero_length_special_tokens() -> None:
    """Special-token predictions must not create zero-length entities."""

    assert token_predictions_to_spans(
        offsets=[(0, 0), (5, 11), (0, 0)],
        labels=["B-EMAIL", "B-PERSON", "I-EMAIL"],
    ) == [DetectedSpan(5, 11, "PERSON")]


def test_token_predictions_reject_mismatched_lengths() -> None:
    """Each predicted label requires one tokenizer offset."""

    with pytest.raises(ValueError, match="offsets and labels"):
        token_predictions_to_spans([(0, 4)], ["O", "O"])


def test_token_predictions_reject_invalid_score() -> None:
    """Model probabilities must remain within the detector score contract."""

    with pytest.raises(ValueError, match="between 0 and 1"):
        token_predictions_to_spans([(0, 4)], ["B-PERSON"], [1.2])
