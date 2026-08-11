"""Load frozen confidence-threshold profiles."""

import json
from collections.abc import Mapping
from pathlib import Path

DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "thresholds.yaml"
)


def load_threshold_profile(
    recall_mode: str,
    *,
    path: Path | None = None,
) -> dict[str, float]:
    """Load one frozen threshold profile.

    Example:
        ``load_threshold_profile("strict")`` returns the
        privacy-first per-entity thresholds.
    """

    if recall_mode not in (
        "balanced",
        "strict",
    ):
        raise ValueError(
            "recall_mode must be 'balanced' or 'strict'"
        )

    config_path = (
        path
        if path is not None
        else DEFAULT_THRESHOLDS_PATH
    )

    raw = json.loads(
        config_path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(raw, dict):
        raise ValueError(
            "threshold config must contain an object"
        )

    profile = raw.get(
        recall_mode,
    )

    if not isinstance(profile, dict):
        raise ValueError(
            f"missing threshold profile: {recall_mode}"
        )

    return _validate_profile(
        profile,
    )


def _validate_profile(
    profile: Mapping[object, object],
) -> dict[str, float]:
    """Validate and normalize one threshold mapping."""

    result: dict[str, float] = {}

    for entity, threshold in profile.items():
        if not isinstance(entity, str) or not entity:
            raise ValueError(
                "threshold entity names must be non-empty strings"
            )

        if (
            isinstance(threshold, bool)
            or not isinstance(
                threshold,
                int | float,
            )
        ):
            raise TypeError(
                f"threshold for {entity} must be numeric"
            )

        value = float(threshold)

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"threshold for {entity} must be between 0 and 1"
            )

        result[entity] = value

    return result