"""Integration tests for the public scrubber orchestration API."""

from pii_scrub import Scrubber
from pii_scrub.config import ScrubberConfig
from pii_scrub.detectors.encoder import EncoderDetector
from pii_scrub.types import DetectedSpan


def test_default_scrubber_redacts_and_restores_structured_pii() -> None:
    text = "Email ana@example.com."
    result = Scrubber().scrub(text)
    assert result.text == "Email [EMAIL_1]."
    assert Scrubber.restore(result.text, result.mapping) == text


def test_scrubber_applies_per_entity_thresholds() -> None:
    detector = EncoderDetector(
        lambda text, entities: [
            DetectedSpan(0, 4, "PERSON", 0.4),
            DetectedSpan(9, 13, "PERSON", 0.9),
        ]
    )
    scrubber = Scrubber(detector, ScrubberConfig(thresholds={"PERSON": 0.5}))
    result = scrubber.scrub("John and Jane")
    assert result.text == "John and [PERSON_1]"
