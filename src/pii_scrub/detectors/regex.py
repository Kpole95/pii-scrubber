"""Deterministic regex baseline for structured identifiers."""

import re
from collections.abc import Mapping

from pii_scrub.types import DetectedSpan

DEFAULT_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "PHONE": re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{6,}\d)(?!\w)"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


class RegexDetector:
    """Detect structured PII using named, independently testable patterns."""

    def __init__(self, patterns: Mapping[str, re.Pattern[str]] = DEFAULT_PATTERNS) -> None:
        """Copy the configured patterns for local matching."""
        self._patterns = dict(patterns)

    def detect(self, text: str, *, entities: set[str] | None = None) -> list[DetectedSpan]:
        """Return non-overlapping regex matches in document order."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        allowed = set(self._patterns) if entities is None else entities
        matches = [
            DetectedSpan(match.start(), match.end(), entity, 1.0)
            for entity, pattern in self._patterns.items()
            if entity in allowed
            for match in pattern.finditer(text)
        ]
        return _resolve(matches)


def _resolve(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    """Prefer longer matches when baseline patterns overlap."""

    selected: list[DetectedSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, -item.length, item.entity_type)):
        if any(span.start < current.end and current.start < span.end for current in selected):
            continue
        selected.append(span)
    return sorted(selected, key=lambda item: (item.start, item.end, item.entity_type))
