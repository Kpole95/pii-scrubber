# tests / unit / research / eval

Unit tests for research evaluation, calibration, metrics, and model-output handling.

## Files

- `__init__.py` — Eval research unit tests. Main defs/classes: No top-level defs or classes.
- `test_ablation.py` — Tests for encoder evaluation ablations. Main defs/classes: `test_merges_person_spans_across_whitespace`, `test_merges_across_punctuation`, `test_does_not_merge_across_words`, `test_does_not_merge_different_entity_types`, `test_empty_predictions_remain_empty`, `test_wrapper_merges_predictor_output`, plus 1 more test.
- `test_baselines.py` — Tests for detector baseline evaluation helpers. Main defs/classes: `FakeDetector`, `test_predict_spans_normalizes_detector_output`, `test_predict_spans_with_regex_detector`, `test_predict_spans_with_presidio_detector`, `test_score_example_uses_shared_metrics`, `test_score_dataset_aggregates_counts`, plus 1 more test.
- `test_calibration.py` — Tests for encoder confidence calibration utilities. Main defs/classes: `_example`, `test_exact_prediction_correctness_requires_exact_match`, `test_calibration_error_uses_exact_correctness`, `test_per_entity_calibration_error_separates_entities`, `test_threshold_grid_includes_zero_and_one`, `test_threshold_grid_rejects_invalid_step`, plus 2 more tests.
- `test_encoder_eval.py` — Tests for encoder evaluation inference helpers. Main defs/classes: `test_prediction_pipeline_decodes_and_merges_duplicate_windows`, `test_prediction_pipeline_supports_entity_filter`, `test_split_words_preserves_source_offsets`, `test_project_offsets_removes_sentencepiece_whitespace_shift`, `test_project_offsets_preserves_overlapping_sentencepiece_pieces`, `test_project_offsets_rejects_invalid_word_id`, plus 1 more test.
- `test_errors.py` — Tests for span-level evaluation error classification. Main defs/classes: `test_exact_match_is_not_an_error`, `test_classifies_complete_miss`, `test_classifies_wrong_type`, `test_classifies_contiguous_split_entity`, `test_classifies_whitespace_separated_split_entity`, `test_classifies_punctuation_separated_split_entity`, plus 5 more tests.
- `test_metrics.py` — Tests for operational span-level evaluation metrics. Main defs/classes: `test_partial_redaction_counts_as_leak`, `test_full_coverage_prevents_leak_even_with_wrong_type`, `test_exact_and_partial_scores_differ_on_boundary_error`, `test_partial_matching_is_one_to_one`, `test_per_entity_recall_separates_types`, `test_over_redaction_rate_counts_only_non_pii_characters`, plus 1 more test.
- `test_qwen_detector.py` — Tests for the Qwen detector adapter. Main defs/classes: `test_qwen_detector_parses_generated_spans`, `test_qwen_detector_filters_requested_entities`, `test_qwen_detector_counts_parse_failure`, `test_qwen_detector_rejects_unsorted_output`, `test_qwen_detector_accumulates_parse_failures`, `test_qwen_detector_enforces_known_entity_types`, plus 1 more test.
- `test_qwen_eval.py` — Tests for strict Qwen generative output parsing. Main defs/classes: `test_parses_valid_span_output`, `test_parses_empty_span_output`, `test_allows_json_surrounding_whitespace`, `test_rejects_markdown_fence`, `test_rejects_wrong_top_level_schema`, `test_rejects_non_integer_offsets`, plus 6 more tests.
- `test_qwen_inference.py` — Tests for Qwen inference helpers. Main defs/classes: `FakeBatch`, `test_generate_qwen_output_uses_chat_template`, `test_generate_qwen_output_accepts_custom_token_limit`.
- `test_report_io.py` — Tests for evaluation report persistence. Main defs/classes: `test_write_report_round_trip`, `test_write_report_is_valid_json`, `test_read_report_rejects_non_object`.
- `test_run_baseline.py` — Tests for reusable baseline reports. Main defs/classes: `PerfectDetector`, `test_evaluate_baseline_returns_report`.
- `test_threshold_profiles.py` — Tests for per-entity threshold profile scoring. Main defs/classes: `_example`, `test_profile_threshold_filters_prediction`, `test_none_score_is_always_kept`, `test_strict_optimizer_prefers_lower_leak`, `test_balanced_optimizer_removes_false_positive`, `test_mismatched_prediction_collections_fail`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
