"""Cached encoder prediction helpers used by profile evaluation scripts."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pii_scrub.calibration import apply_thresholds
from pii_scrub.detectors.encoder import EncoderDetector
from pii_scrub.types import CharacterSpan, DetectedSpan
from research.data.models import DatasetExample
from research.eval.run_baseline import evaluate_baseline

Predictions = tuple[tuple[DetectedSpan, ...], ...]


def predict_examples(
    examples: Sequence[DatasetExample],
    predictor: Callable[[str, set[str] | None], list[DetectedSpan]],
    *,
    entities: set[str] | None,
    progress_every: int,
) -> Predictions:
    """Run encoder inference once for each example and cache the spans."""

    predictions: list[tuple[DetectedSpan, ...]] = []
    total = len(examples)

    for index, example in enumerate(examples, start=1):
        predictions.append(tuple(predictor(example.text, entities)))
        if index == total or index % progress_every == 0:
            print(f"predicted {index:,}/{total:,}", flush=True)

    return tuple(predictions)


def evaluate_cached(
    examples: Sequence[DatasetExample],
    predictions: Predictions,
    *,
    thresholds: dict[str, float] | None = None,
    entities: set[str] | None,
) -> dict[str, object]:
    """Evaluate cached predictions without rerunning encoder inference."""

    index = 0

    def replay(
        text: str,
        requested_entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        """Replay the next cached prediction in dataset order."""

        nonlocal index
        if index >= len(predictions):
            raise RuntimeError("prediction cache exhausted")

        example = examples[index]
        if text != example.text:
            raise ValueError("prediction cache and examples are out of order")

        output = list(predictions[index])
        index += 1
        if thresholds is not None:
            output = apply_thresholds(output, thresholds)
        if requested_entities is not None:
            output = [
                prediction for prediction in output if prediction.entity_type in requested_entities
            ]
        return output

    report = evaluate_baseline(
        EncoderDetector(replay),
        examples,
        entities=entities,
    )
    if index != len(predictions):
        raise RuntimeError("prediction cache was not fully consumed")
    return report


def write_prediction_cache(
    examples: Sequence[DatasetExample],
    predictions: Predictions,
    output: Path,
) -> None:
    """Write normalized examples and scored predictions as JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, (example, example_predictions) in enumerate(
            zip(examples, predictions, strict=True)
        ):
            record = {
                "example_index": index,
                "example": asdict(example),
                "predictions": [asdict(prediction) for prediction in example_predictions],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_prediction_cache(path: Path) -> tuple[tuple[DatasetExample, ...], Predictions]:
    """Load normalized examples and scored predictions from JSONL."""

    examples: list[DatasetExample] = []
    predictions: list[tuple[DetectedSpan, ...]] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record: dict[str, Any] = json.loads(line)
            examples.append(_example_from_cache(record["example"]))
            predictions.append(
                tuple(_prediction_from_cache(item) for item in record["predictions"])
            )

    return tuple(examples), tuple(predictions)


def _example_from_cache(raw: Any) -> DatasetExample:
    """Build one normalized example from a prediction-cache record."""

    if not isinstance(raw, dict):
        raise TypeError("cached example must be a dictionary")

    spans = tuple(
        CharacterSpan(
            start=span["start"],
            end=span["end"],
            entity_type=span["entity_type"],
        )
        for span in raw["spans"]
    )
    return DatasetExample(
        example_id=raw["example_id"],
        source=raw["source"],
        language=raw["language"],
        text=raw["text"],
        spans=spans,
    )


def _prediction_from_cache(raw: Any) -> DetectedSpan:
    """Build one scored prediction from a prediction-cache record."""

    if not isinstance(raw, dict):
        raise TypeError("cached prediction must be a dictionary")
    return DetectedSpan(
        start=raw["start"],
        end=raw["end"],
        entity_type=raw["entity_type"],
        score=raw["score"],
    )
