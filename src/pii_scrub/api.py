"""Public orchestration API for local PII redaction and restoration."""

from dataclasses import dataclass

from pii_scrub.calibration import apply_thresholds
from pii_scrub.config import ScrubberConfig
from pii_scrub.detectors.base import Detector
from pii_scrub.detectors.regex import RegexDetector
from pii_scrub.text.replacement import (
    RestoreEntry,
    replace_spans,
    restore_text,
)
from pii_scrub.threshold_profiles import load_threshold_profile


@dataclass(frozen=True, slots=True)
class ScrubResult:
    """Return redacted text plus the caller-owned restoration mapping."""

    text: str
    mapping: tuple[RestoreEntry, ...]


class Scrubber:
    """Detect, threshold, and replace PII without sending text to a cloud API."""

    def __init__(
        self,
        detector: Detector | None = None,
        config: ScrubberConfig | None = None,
    ) -> None:
        self._detector = detector or RegexDetector()
        self._config = config or ScrubberConfig()

    def scrub(
        self,
        text: str,
        *,
        entities: set[str] | None = None,
    ) -> ScrubResult:
        """Redact requested entities and return a reversible result."""

        spans = self._detector.detect(
            text,
            entities=entities,
        )

        thresholds = self._config.thresholds

        if not thresholds:
            thresholds = load_threshold_profile(
                self._config.recall_mode,
            )

        filtered = apply_thresholds(
            spans,
            thresholds,
        )

        result = replace_spans(
            text,
            filtered,
        )

        return ScrubResult(
            result.text,
            result.mapping,
        )

    @staticmethod
    def restore(
        text: str,
        mapping: tuple[RestoreEntry, ...],
    ) -> str:
        """Restore text using the mapping returned by :meth:`scrub`."""

        return restore_text(
            text,
            mapping,
        )