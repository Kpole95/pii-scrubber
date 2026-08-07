"""Tests for operational span-level evaluation metrics."""

import pytest

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.eval.metrics import (
    exact_span_prf,
    expected_calibration_error,
    leak_rate,
    over_redaction_rate,
    partial_span_prf,
    per_entity_recall,
)


def test_partial_redaction_counts_as_leak() -> None:
    gold = [CharacterSpan(0, 20, "PERSON")]
    predicted = [CharacterSpan(0, 14, "PERSON")]
    assert leak_rate(gold, predicted) == 1.0


def test_full_coverage_prevents_leak_even_with_wrong_type() -> None:
    gold = [CharacterSpan(5, 11, "PERSON")]
    predicted = [CharacterSpan(5, 11, "ADDRESS")]
    assert leak_rate(gold, predicted) == 0.0


def test_exact_and_partial_scores_differ_on_boundary_error() -> None:
    gold = [CharacterSpan(0, 10, "PERSON")]
    predicted = [CharacterSpan(0, 8, "PERSON")]
    assert exact_span_prf(gold, predicted).f1 == 0.0
    assert partial_span_prf(gold, predicted).f1 == 1.0


def test_partial_matching_is_one_to_one() -> None:
    gold = [CharacterSpan(0, 10, "PERSON")]
    predicted = [CharacterSpan(0, 5, "PERSON"), CharacterSpan(5, 10, "PERSON")]
    result = partial_span_prf(gold, predicted)
    assert result.true_positives == 1
    assert result.false_positives == 1


def test_per_entity_recall_separates_types() -> None:
    gold = [CharacterSpan(0, 4, "PERSON"), CharacterSpan(10, 20, "EMAIL")]
    predicted = [CharacterSpan(0, 4, "PERSON")]
    assert per_entity_recall(gold, predicted) == {"EMAIL": 0.0, "PERSON": 1.0}


def test_over_redaction_rate_counts_only_non_pii_characters() -> None:
    gold = [CharacterSpan(0, 4, "PERSON")]
    predicted = [CharacterSpan(0, 6, "PERSON")]
    assert over_redaction_rate(10, gold, predicted) == 2 / 6


def test_ece_uses_score_accuracy_gap() -> None:
    predictions = [
        DetectedSpan(0, 1, "PERSON", 0.9),
        DetectedSpan(2, 3, "PERSON", 0.1),
    ]
    assert expected_calibration_error(predictions, [True, False], bins=2) == pytest.approx(0.1)
