# src / pii_scrub / detectors

Detector interfaces and adapters used by the public scrubber API.

## Files

- `__init__.py` — Detector interfaces and runtime adapters. Main defs/classes: No top-level defs or classes.
- `base.py` — Common detector contract used by the public scrubber API. Main defs/classes: `Detector`.
- `encoder.py` — Dependency-injected runtime adapter for an encoder token classifier. Main defs/classes: `EncoderDetector`.
- `generative.py` — JSON-span adapter for a local generative detector. Main defs/classes: `GenerativeDetector`.
- `presidio.py` — Optional Microsoft Presidio adapter with lazy imports. Main defs/classes: `PresidioDetector`.
- `regex.py` — Deterministic regex baseline for structured identifiers. Main defs/classes: `RegexDetector`, `_resolve`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
