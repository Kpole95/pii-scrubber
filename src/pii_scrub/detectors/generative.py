"""JSON-span adapter for a local generative detector."""

import json
from collections.abc import Callable

from pii_scrub.errors import DetectorError
from pii_scrub.types import DetectedSpan

Generator = Callable[[str], str]


class GenerativeDetector:
    """Parse model-generated JSON spans without allowing text rewriting."""

    def __init__(self, generator: Generator) -> None:
        """Store the injected local generation function."""
        if not callable(generator):
            raise TypeError("generator must be callable")
        self._generator = generator

    def detect(self, text: str, *, entities: set[str] | None = None) -> list[DetectedSpan]:
        """Generate, validate, filter, and sort character spans."""

        try:
            payload = json.loads(self._generator(text))
        except (TypeError, json.JSONDecodeError) as error:
            raise DetectorError("generative detector returned invalid JSON") from error
        if not isinstance(payload, list):
            raise DetectorError("generative detector output must be a JSON list")

        spans: list[DetectedSpan] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise DetectorError(f"generated span at index {index} must be an object")
            try:
                span = DetectedSpan(
                    start=item["start"],
                    end=item["end"],
                    entity_type=item["entity_type"],
                    score=item.get("score"),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise DetectorError(f"generated span at index {index} is invalid") from error
            if span.end > len(text):
                raise DetectorError(f"generated span at index {index} exceeds text length")
            if entities is None or span.entity_type in entities:
                spans.append(span)
        return sorted(spans, key=lambda item: (item.start, item.end, item.entity_type))
