# tests / unit / text / spans

Tests for span models, validation, replacement, and restoration behavior.

## Files

- `__init__.py` — Text spans unit tests. Main defs/classes: No top-level defs or classes.
- `test_models.py` — Tests for window-prediction data models. Main defs/classes: `test_window_span_stores_prediction`, `test_window_span_allows_missing_score`, `test_window_span_converts_integer_score_to_float`, `test_window_span_rejects_negative_window_index`, `test_window_span_rejects_invalid_window_index_type`, `test_window_span_rejects_score_outside_range`, plus 2 more tests.
- `test_replacement.py` — Tests for deterministic replacement and exact restoration. Main defs/classes: `test_replace_and_restore_multiple_entity_types`, `test_repeated_values_get_distinct_placeholders`, `test_replace_rejects_overlapping_spans`, `test_restore_rejects_missing_placeholder`.
- `test_validation.py` — Tests for BIO parsing and numeric-label conversion. Main defs/classes: `test_parse_bio_label_parses_o`, `test_parse_bio_label_parses_entity`, `test_parse_bio_label_rejects_invalid_label`, `test_parse_bio_label_rejects_non_string`, `test_validate_label_mapping_accepts_unique_ids`, `test_validate_label_mapping_rejects_duplicate_ids`, plus 4 more tests.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
