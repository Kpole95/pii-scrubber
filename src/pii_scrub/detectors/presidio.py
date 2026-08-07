"""Optional Microsoft Presidio adapter with lazy imports."""

from pii_scrub.errors import DetectorError
from pii_scrub.types import DetectedSpan


class PresidioDetector:
    """Expose Presidio AnalyzerEngine results through the detector protocol."""

    def __init__(self, analyzer: object | None = None) -> None:
        if analyzer is not None:
            self._analyzer = analyzer
            return
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as error:
            raise DetectorError("install pii-scrubber[presidio] to use Presidio") from error
        self._analyzer = AnalyzerEngine()

    def detect(self, text: str, *, entities: set[str] | None = None) -> list[DetectedSpan]:
        """Analyze English text and normalize Presidio result fields."""

        analyze = getattr(self._analyzer, "analyze", None)
        if not callable(analyze):
            raise DetectorError("analyzer must provide an analyze() method")
        results = analyze(text=text, language="en", entities=sorted(entities) if entities else None)
        spans = [
            DetectedSpan(result.start, result.end, result.entity_type, result.score)
            for result in results
        ]
        return sorted(spans, key=lambda item: (item.start, item.end, item.entity_type))
