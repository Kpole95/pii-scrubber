"""Detector interfaces and runtime adapters."""

from pii_scrub.detectors.base import Detector
from pii_scrub.detectors.encoder import EncoderDetector
from pii_scrub.detectors.generative import GenerativeDetector
from pii_scrub.detectors.presidio import PresidioDetector
from pii_scrub.detectors.regex import RegexDetector

__all__ = [
    "Detector",
    "EncoderDetector",
    "GenerativeDetector",
    "PresidioDetector",
    "RegexDetector",
]
