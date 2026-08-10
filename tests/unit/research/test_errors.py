"""Tests for span-level evaluation error classification."""

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.eval.errors import classify_gold_errors, false_positive_spans


def test_exact_match_is_not_an_error() -> None:
    """Exact same-type predictions should be excluded from errors."""

    gold = [CharacterSpan(0, 10, "PERSON")]
    predictions = [DetectedSpan(0, 10, "PERSON", 0.9)]

    assert classify_gold_errors("John Smith", gold, predictions) == ()


def test_classifies_complete_miss() -> None:
    """Gold spans without overlapping predictions should be misses."""

    errors = classify_gold_errors(
        "John Smith",
        [CharacterSpan(0, 10, "PERSON")],
        [],
    )

    assert errors[0].kind == "miss"


def test_classifies_wrong_type() -> None:
    """Overlapping predictions with another type should be wrong-type errors."""

    errors = classify_gold_errors(
        "John Smith",
        [CharacterSpan(0, 10, "PERSON")],
        [DetectedSpan(0, 10, "LOCATION", 0.8)],
    )

    assert errors[0].kind == "wrong_type"


def test_classifies_contiguous_split_entity() -> None:
    """Adjacent same-type pieces covering one gold span should be split."""

    errors = classify_gold_errors(
        "JohnSmith",
        [CharacterSpan(0, 9, "PERSON")],
        [
            DetectedSpan(0, 4, "PERSON", 0.9),
            DetectedSpan(4, 9, "PERSON", 0.8),
        ],
    )

    assert errors[0].kind == "split"


def test_classifies_whitespace_separated_split_entity() -> None:
    """Separate name tokens covering gold except whitespace should be split."""

    errors = classify_gold_errors(
        "Nadim Ladki",
        [CharacterSpan(0, 11, "PERSON")],
        [
            DetectedSpan(0, 5, "PERSON", 0.9),
            DetectedSpan(6, 11, "PERSON", 0.8),
        ],
    )

    assert errors[0].kind == "split"


def test_classifies_punctuation_separated_split_entity() -> None:
    """Fragments separated only by punctuation should remain one split error."""

    errors = classify_gold_errors(
        "Jean-Claude",
        [CharacterSpan(0, 11, "PERSON")],
        [
            DetectedSpan(0, 4, "PERSON", 0.9),
            DetectedSpan(5, 11, "PERSON", 0.8),
        ],
    )

    assert errors[0].kind == "split"


def test_does_not_join_fragments_across_missing_words() -> None:
    """Fragments with uncovered lexical text should not count as full coverage."""

    errors = classify_gold_errors(
        "John van Smith",
        [CharacterSpan(0, 14, "PERSON")],
        [
            DetectedSpan(0, 4, "PERSON", 0.9),
            DetectedSpan(9, 14, "PERSON", 0.8),
        ],
    )

    assert errors[0].kind == "undersized"


def test_classifies_oversized_prediction() -> None:
    """Predictions containing the gold span should be oversized."""

    errors = classify_gold_errors(
        "Mr. John Smith",
        [CharacterSpan(4, 10, "PERSON")],
        [DetectedSpan(0, 12, "PERSON", 0.9)],
    )

    assert errors[0].kind == "oversized"


def test_classifies_undersized_prediction() -> None:
    """Predictions contained inside gold should be undersized."""

    errors = classify_gold_errors(
        "John Smith",
        [CharacterSpan(0, 10, "PERSON")],
        [DetectedSpan(0, 4, "PERSON", 0.9)],
    )

    assert errors[0].kind == "undersized"


def test_classifies_boundary_shift() -> None:
    """Partial same-type overlap should be classified as a boundary shift."""

    errors = classify_gold_errors(
        "xxJohn Smith",
        [CharacterSpan(5, 12, "PERSON")],
        [DetectedSpan(2, 9, "PERSON", 0.9)],
    )

    assert errors[0].kind == "boundary_shift"


def test_false_positive_requires_no_same_type_overlap() -> None:
    """Predictions without same-type gold overlap should be false positives."""

    gold = [CharacterSpan(0, 10, "PERSON")]
    predictions = [
        DetectedSpan(0, 10, "PERSON", 0.9),
        DetectedSpan(15, 20, "PERSON", 0.8),
        DetectedSpan(0, 10, "LOCATION", 0.7),
    ]

    assert false_positive_spans(gold, predictions) == (
        DetectedSpan(15, 20, "PERSON", 0.8),
        DetectedSpan(0, 10, "LOCATION", 0.7),
    )
