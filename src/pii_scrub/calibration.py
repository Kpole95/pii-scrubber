"""Apply calibrated per-entity confidence thresholds at runtime."""

from collections.abc import Mapping, Sequence

from pii_scrub.types import DetectedSpan


def apply_thresholds(
    spans: Sequence[DetectedSpan], thresholds: Mapping[str, float]
) -> list[DetectedSpan]:
    """Keep spans whose score meets their entity threshold.

    Spans without scores are retained because deterministic detectors such as
    regex baselines do not produce calibrated probabilities.
    """

    return [
        span
        for span in spans
        if span.score is None or span.score >= thresholds.get(span.entity_type, 0.0)
    ]
