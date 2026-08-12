# research / eval

Evaluation metrics, calibration, threshold selection, error analysis, and model inference helpers.

## Files

- `__init__.py` — Span-level evaluation and robustness experiments. Main defs/classes: No top-level defs or classes.
- `ablation.py` — Prediction post-processing ablations for encoder evaluation. Main defs/classes: `merge_separated_same_type`, `wrap_merge_ablation`.
- `analyze_errors.py` — Analyze encoder span errors on normalized evaluation examples. Main defs/classes: `main`, `_parser`.
- `baselines.py` — Helpers for evaluating runtime detectors as research baselines. Main defs/classes: `BaselineScore`, `DatasetScore`, `score_dataset`, `_dataset_entity_recall`, `_entity_recall`, `_merge_prf`, `_dataset_leak_rate`, `predict_spans`, `_dataset_over_redaction`, `score_example`.
- `calibration.py` — Calibration utilities for scored PII span predictions. Main defs/classes: `ThresholdScore`, `exact_prediction_correctness`, `calibration_error`, `per_entity_calibration_error`, `threshold_grid`, `sweep_thresholds`, `_entity_calibration_error`, `_score_threshold`, `_validate_predictions`.
- `encoder.py` — Load and run the trained encoder for span-level evaluation. Main defs/classes: `load_encoder_predictor`, `_split_words`, `_project_offsets`.
- `encoder_profiles.py` — Cached encoder prediction helpers used by profile evaluation scripts. Main defs/classes: `predict_examples`, `evaluate_cached`, `write_prediction_cache`, `load_prediction_cache`, `_example_from_cache`, `_prediction_from_cache`.
- `errors.py` — Classify span-level prediction errors for evaluation analysis. Main defs/classes: `SpanError`, `classify_gold_errors`, `false_positive_spans`, `_classify`, `_same_span`, `_overlaps`, `_covers_gold`, `_overlap_size`.
- `inspect_merge_regressions.py` — Inspect exact matches broken by the PERSON merge ablation. Main defs/classes: `main`, `_parser`.
- `metrics.py` — Span-level metrics for PII redaction systems. Main defs/classes: `PrecisionRecallF1`, `leak_rate`, `exact_span_prf`, `partial_span_prf`, `per_entity_recall`, `over_redaction_rate`, `expected_calibration_error`, `_fully_covered`, `_overlap_length`, `_covered_characters`, `_validate_spans`, `_validate_bounds`, plus 1 more def.
- `qwen.py` — Parse generative Qwen PII predictions into detected spans. Main defs/classes: `parse_qwen_output`, `_parse_span`, `_validate_order`.
- `qwen_detector.py` — Detector adapter for Qwen generative PII predictions. Main defs/classes: `QwenDetector`.
- `qwen_inference.py` — Inference helpers for the Qwen LoRA PII detector. Main defs/classes: `generate_qwen_output`, `load_qwen_generator`.
- `report_io.py` — Read and write evaluation reports. Main defs/classes: `write_report`, `read_report`.
- `run_baseline.py` — Build serializable reports for detector baselines. Main defs/classes: `evaluate_baseline`, `_report`, `_prf_report`.
- `threshold_metrics.py` — Shared metric helpers for confidence-threshold evaluation. Main defs/classes: `ThresholdMetrics`, `score_filtered_predictions`, `_as_character_span`, `_gold_characters`, `_ratio`, `_f1`.
- `threshold_profiles.py` — Select and score per-entity confidence-threshold profiles. Main defs/classes: `ThresholdProfileScore`, `score_threshold_profile`, `optimize_threshold_profile`, `_optimize_pass`, `_passes_threshold`, `_is_better`, `_validate_predictions`, `_validate_thresholds`, `_validate_threshold_value`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
