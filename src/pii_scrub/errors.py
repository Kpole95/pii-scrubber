"""Package-specific exceptions exposed by :mod:`pii_scrub`."""


class PiiScrubError(Exception):
    """Base exception for recoverable PII Scrubber failures."""


class ConfigurationError(PiiScrubError):
    """Raised when runtime configuration is invalid."""


class DetectorError(PiiScrubError):
    """Raised when a detector cannot produce trustworthy spans."""


class InvalidSpanError(PiiScrubError, ValueError):
    """Raised when character offsets are malformed or conflict."""


class RestoreError(PiiScrubError):
    """Raised when redacted text cannot be restored safely."""
