# tests / contract

Contract tests that make sure detector implementations follow the shared detector interface.

## Files

- `__init__.py` — Contract tests for detector interfaces and shared behavior. Main defs/classes: No top-level defs or classes.
- `test_detectors.py` — Contract tests shared by lightweight detector implementations. Main defs/classes: `test_regex_detector_returns_document_offsets`, `test_regex_detector_respects_entity_filter`, `test_encoder_adapter_sorts_predictions`, `test_generative_adapter_parses_json_spans`, `test_generative_adapter_rejects_rewritten_text`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
