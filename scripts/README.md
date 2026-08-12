# scripts

Small command-line entry points for checks, evaluation, calibration, and experiment reporting.

## Files

- `calibrate_encoder.py` — Calibrate encoder confidence scores on a normalized validation split. Main defs/classes: `main`, `_select_split`, `_parser`.
- `check_encoder_data.py` — Smoke-check encoder preparation with a real tokenizer. Main defs/classes: `main`.
- `check_encoder_dataset.py` — Inspect encoder preparation on the real Ai4Privacy artifact. Main defs/classes: `main`.
- `check_qwen_dataset.py` — Inspect real Qwen chat-token lengths for normalized training data. Main defs/classes: `main`, `_report_split`, `_print_stats`, `_percentile`, `_parser`.
- `check_qwen_lora.py` — Inspect the Stage 11 Qwen LoRA parameter footprint. Main defs/classes: `main`, `_parser`.
- `check_structure.py` — Validate repository structure, module size, and documentation rules. Main defs/classes: `main`, `project_modules`, `check_tracked_generated_files`, `check_forbidden_modules`, `check_module_sizes`, `check_docstrings`, `check_runtime_imports`.
- `evaluate_encoder_profiles.py` — Evaluate frozen encoder threshold profiles from one inference pass. Main defs/classes: `main`, `_load_examples`, `_print_summary`, `_parser`.
- `ood_progress.py` — Print progress for the hand-labeled OOD benchmark. Main defs/classes: `main`.
- `run_evaluation.py` — Run one detector evaluation from CLI arguments. Main defs/classes: `main`, `_build_detector`, `_load_examples`, `_parser`.
- `run_stage13.py` — Run the Stage 13 robustness and memorization audit. Main defs/classes: `main`, `transform_example`, `robustness_audit`, `long_context_audit`, `memorization_audit`, `_duplicate_stats`, `_pii_value_overlap`, `_predict_cache`, `_cached_detector`, `_metrics`, `_delta`, `_prefix_context`, plus 3 more defs.
- `select_encoder_thresholds.py` — Select frozen encoder threshold profiles from validation predictions. Main defs/classes: `main`, `_selection_report`, `_score_dict`, `_write_threshold_config`, `_print_profile`, `_parser`.
- `summarize_baselines.py` — Write a Markdown summary for baseline evaluation reports. Main defs/classes: `main`, `_row`, `_markdown`, `_number`, `_parser`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
