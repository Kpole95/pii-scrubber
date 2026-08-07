"""Common detector contract used by the public scrubber API."""

from collections.abc import Sequence
from typing import Protocol

from pii_scrub.types import DetectedSpan


class Detector(Protocol):
    """Detect PII and return original-text character offsets."""

    def detect(self, text: str, *, entities: set[str] | None = None) -> Sequence[DetectedSpan]:
        """Return sorted candidate spans for ``text``."""
