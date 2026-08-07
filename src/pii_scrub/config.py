"""Validated immutable runtime configuration."""

from dataclasses import dataclass, field

from pii_scrub.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ScrubberConfig:
    """Configure detector thresholds and redaction mode."""

    recall_mode: str = "balanced"
    thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.recall_mode not in {"balanced", "strict"}:
            raise ConfigurationError("recall_mode must be 'balanced' or 'strict'")
        for entity, threshold in self.thresholds.items():
            if not isinstance(entity, str) or not entity:
                raise ConfigurationError("threshold entity names must be non-empty strings")
            if isinstance(threshold, bool) or not isinstance(threshold, int | float):
                raise ConfigurationError(f"threshold for {entity} must be numeric")
            if not 0.0 <= float(threshold) <= 1.0:
                raise ConfigurationError(f"threshold for {entity} must be between 0 and 1")
