"""Tests for the Stage 13 robustness and memorization audit."""

from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from scripts.run_stage13 import (
    long_context_audit,
    memorization_audit,
    robustness_audit,
    transform_example,
)


def _example(example_id: str, text: str, value: str = "Alice") -> DatasetExample:
    start = text.index(value)
    return DatasetExample(
        example_id,
        text,
        (CharacterSpan(start, start + len(value), "PERSON"),),
        "test",
        "en",
    )


def _alice_predictor(text: str, entities: set[str] | None) -> list[DetectedSpan]:
    start = text.casefold().index("alice")
    span = DetectedSpan(start, start + 5, "PERSON", 0.9)
    return [span] if entities is None or "PERSON" in entities else []


def test_transform_example_remaps_offsets_after_expansion() -> None:
    example = _example("one", "Call Alice now")
    changed = transform_example(
        example,
        "double",
        lambda char: "  " if char == " " else char,
    )

    assert changed.text == "Call  Alice  now"
    assert changed.spans == (CharacterSpan(6, 11, "PERSON"),)
    assert changed.spans[0].extract(changed.text) == "Alice"


def test_robustness_audit_keeps_perfect_case_under_simple_predictor() -> None:
    report = robustness_audit(_alice_predictor, (_example("one", "Call Alice - it's fine"),))

    baseline = report["baseline"]
    assert isinstance(baseline, dict)
    assert baseline["exact_f1"] == 1.0
    assert baseline["leak_rate"] == 0.0

    perturbations = report["perturbations"]
    assert isinstance(perturbations, dict)
    lowercase = perturbations["lowercase"]
    assert isinstance(lowercase, dict)
    assert lowercase["delta"] == {
        "exact_f1": 0.0,
        "partial_f1": 0.0,
        "leak_rate": 0.0,
        "over_redaction_rate": 0.0,
    }


def test_long_context_audit_preserves_shifted_gold() -> None:
    report = long_context_audit(_alice_predictor, (_example("one", "Call Alice now"),))

    variants = report["variants"]
    assert isinstance(variants, dict)
    for value in variants.values():
        assert isinstance(value, dict)
        metrics = value["metrics"]
        assert isinstance(metrics, dict)
        assert metrics["exact_f1"] == 1.0
        assert metrics["leak_rate"] == 0.0


def test_memorization_audit_counts_exact_normalized_and_value_overlap() -> None:
    train = (_example("train", "Call Alice now"),)
    validation = (_example("val", "Call Alice now"),)
    test = (_example("test", "  CALL  ALICE  NOW  ", value="ALICE"),)
    ood = (_example("ood", "Meet Bob today", value="Bob"),)

    report = memorization_audit(train, validation, test, ood)

    exact = report["exact_duplicates_from_train"]
    normalized = report["normalized_duplicates_from_train"]
    overlap = report["pii_value_overlap_from_train"]
    assert isinstance(exact, dict)
    assert isinstance(normalized, dict)
    assert isinstance(overlap, dict)
    assert exact["validation"]["matching_examples"] == 1
    assert exact["test"]["matching_examples"] == 0
    assert normalized["test"]["matching_examples"] == 1
    assert overlap["test"]["PERSON"]["seen_in_train"] == 1
    assert overlap["ood"]["PERSON"]["seen_in_train"] == 0
