# src / pii_scrub

The installable runtime package for PII detection, redaction, restoration, and configuration.

## Folders

- `detectors/` — Detector implementations and adapters.
- `text/` — Text and span processing primitives.

## Files

- `__init__.py` — Local PII detection and reversible redaction. Main defs/classes: No top-level defs or classes.
- `api.py` — Public orchestration API for local PII redaction and restoration. Main defs/classes: `ScrubResult`, `Scrubber`.
- `calibration.py` — Apply calibrated per-entity confidence thresholds at runtime. Main defs/classes: `apply_thresholds`.
- `cli.py` — Command-line interface for local regex-based redaction. Main defs/classes: `build_parser`, `main`.
- `config.py` — Validated immutable runtime configuration. Main defs/classes: `ScrubberConfig`.
- `errors.py` — Package-specific exceptions exposed by :mod:`pii_scrub`. Main defs/classes: `PiiScrubError`, `ConfigurationError`, `DetectorError`, `InvalidSpanError`, `RestoreError`.
- `py.typed` — Typing marker showing that the installed package includes type information.
- `threshold_profiles.py` — Load frozen confidence-threshold profiles. Main defs/classes: `load_threshold_profile`, `_validate_profile`.
- `types.py` — Immutable domain models shared by runtime components. Main defs/classes: `CharacterSpan`, `DetectedSpan`, `WordOffset`, `AlignedExample`, `_is_offset`, `WindowSpan`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
