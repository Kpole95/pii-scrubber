"""Dependency-injected runtime adapter for an encoder token classifier."""

from collections.abc import Callable, Sequence

from pii_scrub.types import DetectedSpan

Predictor = Callable[[str, set[str] | None], Sequence[DetectedSpan]]


class EncoderDetector:
    """Adapt a loaded encoder predictor to the shared detector contract.

    Model loading and training remain outside this module. This keeps the
    runtime adapter testable and prevents heavy imports at package import time.
    """

    def __init__(self, predictor: Predictor) -> None:
        """Store the injected encoder prediction function."""
        if not callable(predictor):
            raise TypeError("predictor must be callable")
        self._predictor = predictor

    def detect(self, text: str, *, entities: set[str] | None = None) -> list[DetectedSpan]:
        """Validate and sort spans returned by the injected predictor."""

        spans = list(self._predictor(text, entities))
        if any(not isinstance(span, DetectedSpan) for span in spans):
            raise TypeError("encoder predictor must return DetectedSpan objects")
        return sorted(spans, key=lambda item: (item.start, item.end, item.entity_type))
