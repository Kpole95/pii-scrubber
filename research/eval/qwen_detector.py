"""Detector adapter for Qwen generative PII predictions."""

from collections.abc import Callable

from pii_scrub.types import DetectedSpan
from research.eval.qwen import parse_qwen_output

QwenGenerator = Callable[[str], str]


class QwenDetector:
    """Adapt Qwen JSON generation to the shared detector contract."""

    def __init__(
        self,
        generator: QwenGenerator,
        *,
        entity_types: set[str] | None = None,
    ) -> None:
        if not callable(generator):
            raise TypeError("generator must be callable")

        self._generator = generator
        self._entity_types = entity_types
        self._parse_failures = 0

    @property
    def parse_failures(self) -> int:
        """Return the number of invalid model outputs seen so far."""

        return self._parse_failures

    def detect(
        self,
        text: str,
        *,
        entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        """Generate, parse, filter, and return PII spans for one text."""

        output = self._generator(text)

        try:
            spans = parse_qwen_output(
                text,
                output,
                entity_types=self._entity_types,
            )
        except ValueError:
            self._parse_failures += 1
            return []

        if entities is not None:
            spans = [span for span in spans if span.entity_type in entities]

        return spans
