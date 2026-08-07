"""Contract tests shared by lightweight detector implementations."""

import pytest

from pii_scrub.detectors.encoder import EncoderDetector
from pii_scrub.detectors.generative import GenerativeDetector
from pii_scrub.detectors.regex import RegexDetector
from pii_scrub.errors import DetectorError
from pii_scrub.types import DetectedSpan


def test_regex_detector_returns_document_offsets() -> None:
    text = "Email ana@example.com or call +44 7700 900123."
    spans = RegexDetector().detect(text)
    assert [span.entity_type for span in spans] == ["EMAIL", "PHONE"]
    assert [span.extract(text) for span in spans] == ["ana@example.com", "+44 7700 900123"]


def test_regex_detector_respects_entity_filter() -> None:
    spans = RegexDetector().detect("Email ana@example.com", entities={"PHONE"})
    assert spans == []


def test_encoder_adapter_sorts_predictions() -> None:
    detector = EncoderDetector(
        lambda text, entities: [
            DetectedSpan(10, 14, "PERSON", 0.8),
            DetectedSpan(0, 4, "PERSON", 0.9),
        ]
    )
    assert [span.start for span in detector.detect("John sees Jane")] == [0, 10]


def test_generative_adapter_parses_json_spans() -> None:
    detector = GenerativeDetector(
        lambda text: '[{"start": 0, "end": 4, "entity_type": "PERSON", "score": 0.9}]'
    )
    assert detector.detect("John") == [DetectedSpan(0, 4, "PERSON", 0.9)]


def test_generative_adapter_rejects_rewritten_text() -> None:
    detector = GenerativeDetector(lambda text: '"[PERSON]"')
    with pytest.raises(DetectorError, match="JSON list"):
        detector.detect("John")
