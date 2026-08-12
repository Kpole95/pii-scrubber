# tests / unit / scripts

Unit tests for repository and experiment command-line scripts.

## Files

- `test_check_qwen_dataset.py` — Tests for Qwen dataset inspection helpers. Main defs/classes: `test_percentile_minimum`, `test_percentile_median`, `test_percentile_maximum`, `test_percentile_rejects_empty_values`, `test_percentile_rejects_invalid_fraction`.
- `test_evaluate_encoder_profiles.py` — Tests for cached frozen-profile encoder evaluation. Main defs/classes: `_example`, `test_raw_cache_keeps_prediction`, `test_thresholded_cache_can_remove_prediction`, `test_entity_filter_is_preserved`.
- `test_ood_progress.py` — Tests for the OOD progress script. Main defs/classes: `test_prints_ood_progress`.
- `test_run_evaluation.py` — Tests for the detector evaluation CLI. Main defs/classes: `test_parser_accepts_required_arguments`, `test_parser_accepts_artifact_split`, `test_parser_accepts_conll_source`, `test_parser_accepts_encoder_model`, `test_encoder_requires_model_path`, `test_parser_accepts_person_merge_ablation`, plus 1 more test.
- `test_run_stage13.py` — Tests for the Stage 13 robustness and memorization audit. Main defs/classes: `_example`, `_alice_predictor`, `test_transform_example_remaps_offsets_after_expansion`, `test_robustness_audit_keeps_perfect_case_under_simple_predictor`, `test_long_context_audit_preserves_shifted_gold`, `test_memorization_audit_counts_exact_normalized_and_value_overlap`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
