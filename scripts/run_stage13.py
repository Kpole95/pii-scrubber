"""Run the Stage 13 robustness and memorization audit."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pii_scrub.detectors.encoder import EncoderDetector
from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.artifacts import load_dataset_split
from research.data.models import DatasetExample
from research.data.ood_io import load_ood_jsonl
from research.eval.encoder import EncoderPredictor, load_encoder_predictor
from research.eval.run_baseline import evaluate_baseline

CharTransform = Callable[[str], str]
Report = dict[str, object]
TEST_LIMIT = 200
WINDOW_LIMIT = 20
PROGRESS_EVERY = 50


def main() -> None:
    """Run the frozen Stage 13 contract and write one JSON report."""
    args = _parser().parse_args()
    artifact = load_dataset_split(args.artifact)
    ood = load_ood_jsonl(args.ood)
    output: Report = {
        "contract": {
            "reference_policy": "raw_encoder",
            "threshold_retuning": False,
            "near_duplicate_search": "deferred_unless_duplicate_screen_flags_risk",
        },
        "memorization": memorization_audit(
            artifact.split.train, artifact.split.validation, artifact.split.test, ood
        ),
    }
    if not args.memorization_only:
        predictor = load_encoder_predictor(args.model_path, device=args.device)
        output["robustness"] = {
            "ai4privacy_test": robustness_audit(predictor, artifact.split.test[:TEST_LIMIT]),
            "ood": robustness_audit(predictor, ood),
            "long_context_ood": long_context_audit(predictor, ood[:WINDOW_LIMIT]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def transform_example(
    example: DatasetExample,
    name: str,
    transform: CharTransform,
) -> DatasetExample:
    """Apply a character transform while preserving exact gold boundaries."""
    parts: list[str] = []
    boundaries = [0]
    for char in example.text:
        replacement = transform(char)
        parts.append(replacement)
        boundaries.append(boundaries[-1] + len(replacement))
    spans = tuple(
        CharacterSpan(boundaries[span.start], boundaries[span.end], span.entity_type)
        for span in example.spans
    )
    return DatasetExample(
        f"{example.example_id}::{name}",
        "".join(parts),
        spans,
        f"{example.source}::robustness::{name}",
        example.language,
    )


def robustness_audit(
    predictor: EncoderPredictor,
    examples: Sequence[DatasetExample],
) -> Report:
    """Measure fixed perturbations against matching unmodified subsets."""
    base_cache = _predict_cache(predictor, examples, "base")
    base = _metrics(evaluate_baseline(_cached_detector(base_cache), examples))
    families: Report = {}
    for name, transform in _perturbations().items():
        pairs = []
        for example in examples:
            changed_example = transform_example(example, name, transform)
            if changed_example.text != example.text:
                pairs.append((example, changed_example))
        originals = tuple(pair[0] for pair in pairs)
        perturbed = tuple(pair[1] for pair in pairs)
        if not perturbed:
            families[name] = {"changed_examples": 0}
            continue
        changed_cache = _predict_cache(predictor, perturbed, name)
        before = _metrics(evaluate_baseline(_cached_detector(base_cache), originals))
        after = _metrics(evaluate_baseline(_cached_detector(changed_cache), perturbed))
        families[name] = {
            "changed_examples": len(perturbed),
            "baseline_same_subset": before,
            "perturbed": after,
            "delta": _delta(after, before),
        }
    return {"examples": len(examples), "baseline": base, "perturbations": families}


def long_context_audit(
    predictor: EncoderPredictor,
    examples: Sequence[DatasetExample],
) -> Report:
    """Push labeled examples near and beyond the encoder window boundary."""
    selected = tuple(example for example in examples if example.spans)
    if not selected:
        return {"examples": 0, "variants": {}}
    base_cache = _predict_cache(predictor, selected, "window-base")
    before = _metrics(evaluate_baseline(_cached_detector(base_cache), selected))
    variants: Report = {}
    for words in (480, 560):
        changed = tuple(_prefix_context(example, words) for example in selected)
        cache = _predict_cache(predictor, changed, f"prefix-{words}")
        after = _metrics(evaluate_baseline(_cached_detector(cache), changed))
        variants[f"prefix_{words}_words"] = {"metrics": after, "delta": _delta(after, before)}
    return {"examples": len(selected), "baseline": before, "variants": variants}


def memorization_audit(
    train: Sequence[DatasetExample],
    validation: Sequence[DatasetExample],
    test: Sequence[DatasetExample],
    ood: Sequence[DatasetExample],
) -> Report:
    """Screen for exact/normalized text reuse and repeated PII values."""
    targets = {"validation": validation, "test": test, "ood": ood}
    return {
        "exact_duplicates_from_train": {
            name: _duplicate_stats(train, rows, lambda item: item.text)
            for name, rows in targets.items()
        },
        "normalized_duplicates_from_train": {
            name: _duplicate_stats(train, rows, lambda item: _normalize(item.text))
            for name, rows in targets.items()
        },
        "pii_value_overlap_from_train": _pii_value_overlap(train, targets),
    }


def _duplicate_stats(
    train: Sequence[DatasetExample],
    target: Sequence[DatasetExample],
    key: Callable[[DatasetExample], str],
) -> Report:
    reference = {key(example) for example in train}
    matches = [example.example_id for example in target if key(example) in reference]
    return {
        "target_examples": len(target),
        "matching_examples": len(matches),
        "rate": len(matches) / len(target) if target else 0.0,
        "example_ids": matches[:25],
    }


def _pii_value_overlap(
    train: Sequence[DatasetExample],
    targets: dict[str, Sequence[DatasetExample]],
) -> Report:
    known: dict[str, set[str]] = defaultdict(set)
    for example in train:
        for span in example.spans:
            known[span.entity_type].add(_normalize(span.extract(example.text)))
    output: Report = {}
    for name, rows in targets.items():
        totals: Counter[str] = Counter()
        repeated: Counter[str] = Counter()
        for example in rows:
            for span in example.spans:
                totals[span.entity_type] += 1
                value = _normalize(span.extract(example.text))
                if value in known.get(span.entity_type, set()):
                    repeated[span.entity_type] += 1
        output[name] = {
            entity: {
                "gold_spans": totals[entity],
                "seen_in_train": repeated[entity],
                "rate": repeated[entity] / totals[entity],
            }
            for entity in sorted(totals)
        }
    return output


def _predict_cache(
    predictor: EncoderPredictor,
    examples: Sequence[DatasetExample],
    label: str,
) -> dict[str, list[DetectedSpan]]:
    cache: dict[str, list[DetectedSpan]] = {}
    for index, example in enumerate(examples, 1):
        cache[example.text] = predictor(example.text, None)
        if index % PROGRESS_EVERY == 0 or index == len(examples):
            print(f"{label}: {index:,}/{len(examples):,}")
    return cache


def _cached_detector(cache: dict[str, list[DetectedSpan]]) -> EncoderDetector:
    def predict(text: str, entities: set[str] | None) -> list[DetectedSpan]:
        spans = cache[text]
        if entities is None:
            return list(spans)
        return [span for span in spans if span.entity_type in entities]

    return EncoderDetector(predict)


def _metrics(report: Report) -> dict[str, float]:
    exact = cast(dict[str, object], report["exact"])
    partial = cast(dict[str, object], report["partial"])
    return {
        "exact_f1": float(cast(float, exact["f1"])),
        "partial_f1": float(cast(float, partial["f1"])),
        "leak_rate": float(cast(float, report["leak_rate"])),
        "over_redaction_rate": float(cast(float, report["over_redaction_rate"])),
    }


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in before}


def _prefix_context(example: DatasetExample, words: int) -> DatasetExample:
    prefix = "context " * words
    offset = len(prefix)
    spans = tuple(
        CharacterSpan(span.start + offset, span.end + offset, span.entity_type)
        for span in example.spans
    )
    return DatasetExample(
        f"{example.example_id}::prefix-{words}",
        prefix + example.text,
        spans,
        f"{example.source}::robustness::prefix-{words}",
        example.language,
    )


def _perturbations() -> dict[str, CharTransform]:
    return {
        "lowercase": str.lower,
        "uppercase": str.upper,
        "double_spaces": lambda char: "  " if char == " " else char,
        "nbsp_spaces": lambda char: "\u00a0" if char == " " else char,
        "unicode_dashes": lambda char: "\u2013" if char == "-" else char,
        "curly_apostrophes": lambda char: "\u2019" if char == "'" else char,
    }


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run Stage 13 robustness and memorization audits.")
    parser.add_argument("--model-path", type=Path, default=Path("research/results/encoder"))
    parser.add_argument("--artifact", type=Path, default=Path("research/data_artifacts/ai4privacy"))
    parser.add_argument("--ood", type=Path, default=Path("research/labeled_ood/examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("research/results/stage13.json"))
    parser.add_argument("--device")
    parser.add_argument("--memorization-only", action="store_true")
    return parser


if __name__ == "__main__":
    main()
