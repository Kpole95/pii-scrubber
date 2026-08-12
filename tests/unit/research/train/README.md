# tests / unit / research / train

Unit tests for encoder and Qwen training data, tokenization, models, and batching.

## Files

- `__init__.py` — Train research unit tests. Main defs/classes: No top-level defs or classes.
- `test_encoder_data.py` — Tests for encoder training-data preparation. Main defs/classes: `_tokenizer`, `test_builds_bio_for_multiword_entity`, `test_splits_at_entity_boundary`, `test_handles_entity_inside_text_piece`, `test_builds_deterministic_label_mapping`, `test_window_example_keeps_short_example_whole`, plus 4 more tests.
- `test_qwen_collator.py` — Tests for Qwen causal-LM batch collation. Main defs/classes: `_record`, `test_pads_variable_length_records`, `test_padding_is_ignored_by_loss`, `test_preserves_existing_prompt_mask`, `test_returns_long_tensors`, `test_rejects_empty_batch`, plus 4 more tests.
- `test_qwen_data.py` — Tests for generative Qwen training examples. Main defs/classes: `_example`, `test_formats_compact_span_json`, `test_formats_multiple_spans_in_source_order`, `test_formats_no_pii_as_empty_span_list`, `test_preserves_unicode_text`, `test_builds_system_user_assistant_messages`, plus 3 more tests.
- `test_qwen_dataset.py` — Tests for Qwen training dataset preparation. Main defs/classes: `FakeTokenizer`, `_example`, `test_prepares_training_records`, `test_skips_overlong_examples_and_counts_them`, `test_preserves_input_order`, `test_rejects_non_positive_max_length`.
- `test_qwen_model.py` — Tests for Qwen LoRA model helpers. Main defs/classes: `test_rejects_non_positive_lora_rank`, `test_rejects_non_positive_lora_alpha`, `test_rejects_invalid_lora_dropout`, `test_rejects_empty_target_modules`, `test_builds_lora_model`, `test_counts_trainable_parameters`.
- `test_qwen_tokens.py` — Tests for Qwen chat tokenization and assistant-only loss masking. Main defs/classes: `FakeChatTokenizer`, `_messages`, `test_masks_prompt_tokens`, `test_keeps_assistant_response_tokens`, `test_input_and_label_lengths_match`, `test_attention_mask_marks_every_real_token`, plus 5 more tests.
- `test_qwen_train.py` — Tests for Qwen LoRA Trainer configuration. Main defs/classes: `_config`, `test_record_dataset_preserves_records`, `test_diagnostic_arguments_use_one_step`, `test_diagnostic_disables_mixed_precision_on_cpu`, `test_fp16_configuration`, `test_training_dtype_uses_float32_without_cuda`, plus 2 more tests.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
