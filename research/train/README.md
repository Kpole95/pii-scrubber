# research / train

Training and data-preparation code for the encoder and Qwen LoRA experiments.

## Files

- `__init__.py` — Training entry points added only when model contracts are implemented. Main defs/classes: No top-level defs or classes.
- `encoder.py` — Train the DeBERTa PII token classifier. Main defs/classes: `main`, `_parser`.
- `encoder_data.py` — Prepare normalized examples for encoder token classification. Main defs/classes: `words_and_bio`, `window_example`, `build_label_mapping`, `prepare_encoder_example`, `_word_offsets`, `_token_count`, `_largest_end`, `_overlap_start`, `_bio_label`, `prepare_encoder_records`.
- `qwen.py` — Train Qwen2.5 with LoRA for generative PII span extraction. Main defs/classes: `QwenRecordDataset`, `main`, `_training_dtype`, `_training_arguments`, `_parser`.
- `qwen_collator.py` — Batch variable-length Qwen training records for causal-LM training. Main defs/classes: `collate_qwen_records`, `_validate_record`.
- `qwen_data.py` — Prepare deterministic chat examples for generative PII training. Main defs/classes: `QwenMessage`, `format_qwen_target`, `build_qwen_messages`.
- `qwen_dataset.py` — Prepare normalized examples for Qwen LoRA training. Main defs/classes: `QwenTrainingRecord`, `prepare_qwen_dataset`.
- `qwen_model.py` — Build the Qwen causal LM with its Stage 11 LoRA adapter. Main defs/classes: `load_qwen_lora_model`, `trainable_parameter_counts`.
- `qwen_tokens.py` — Tokenize Qwen chat examples with assistant-only training labels. Main defs/classes: `TokenizedText`, `ChatTokenizer`, `tokenize_qwen_messages`, `_validate_messages`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
