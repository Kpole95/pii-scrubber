"""Optional Microsoft Presidio adapter with lazy imports."""

from pii_scrub.errors import DetectorError
from pii_scrub.types import DetectedSpan

ENTITY_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "BANK_ACCOUNT",
    "IP_ADDRESS": "IP_ADDRESS",
    "US_SSN": "SOCIAL_SECURITY_NUMBER",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
}


class PresidioDetector:
    """Expose Presidio results through the detector protocol."""

    def __init__(self, analyzer: object | None = None) -> None:
        """Use a provided analyzer or create Presidio lazily."""
        if analyzer is not None:
            self._analyzer = analyzer
            return

        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as error:
            raise DetectorError("install pii-scrubber[presidio] to use Presidio") from error

        self._analyzer = AnalyzerEngine()

    def detect(
        self,
        text: str,
        *,
        entities: set[str] | None = None,
    ) -> list[DetectedSpan]:
        """Detect PII and normalize Presidio entity labels.

        Example:
            ``EMAIL_ADDRESS`` becomes project label ``EMAIL``.
        """
        analyze = getattr(self._analyzer, "analyze", None)

        if not callable(analyze):
            raise DetectorError("analyzer must provide an analyze() method")

        results = analyze(
            text=text,
            language="en",
            entities=sorted(entities) if entities else None,
        )

        spans = [
            DetectedSpan(
                result.start,
                result.end,
                ENTITY_MAP.get(result.entity_type, result.entity_type),
                result.score,
            )
            for result in results
        ]

        return sorted(
            spans,
            key=lambda span: (span.start, span.end, span.entity_type),
        )
