# research / data

Dataset loaders, normalization, serialization, splitting, artifacts, and dataset statistics.

## Files

- `__init__.py` — Public research-data API. Main defs/classes: No top-level defs or classes.
- `ai4privacy.py` — Convert Ai4Privacy records into the project's normalized dataset format. Main defs/classes: `load_ai4privacy_record`, `_parse_span`, `_required`.
- `artifact_models.py` — Immutable models for persisted dataset split artifacts. Main defs/classes: `DatasetManifest`, `LoadedDatasetArtifact`.
- `artifacts.py` — Persist complete normalized dataset splits with a versioned manifest. Main defs/classes: `save_dataset_split`, `load_dataset_split`, `dataset_manifest_to_dict`, `dataset_manifest_from_dict`, `_directory`, `_required_text`, `_required_int`.
- `conll.py` — Convert CoNLL-2003 rows into PERSON-only OOD dataset examples. Main defs/classes: `load_conll2003_record`, `_tokens`, `_tags`, `_detokenize`, `_person_spans`, `_record_id`.
- `gretel.py` — Normalize records from the Gretel synthetic-finance PII schema. Main defs/classes: `load_gretel_finance_record`, `_decode_spans`, `_parse_annotation`, `_required_text`, `_required_identifier`, `_required_annotation_text`, `_required_int`.
- `huggingface.py` — Load raw records from Hugging Face datasets. Main defs/classes: `load_huggingface_records`, `_validate_text`.
- `loaders.py` — Load normalized dataset records with shared error reporting. Main defs/classes: `load_records`, `load_ai4privacy_records`, `load_conll2003_records`, `load_openpii_records`, `load_gretel_finance_records`.
- `models.py` — Internal data models shared by every dataset loader. Main defs/classes: `DatasetExample`, `RejectedRecord`, `DatasetLoadReport`.
- `normalize.py` — Normalize dataset-specific PII labels into one shared taxonomy. Main defs/classes: `normalize_entity_label`.
- `ood.py` — Load manually labeled out-of-distribution examples. Main defs/classes: `load_ood_record`, `_span`, `_required`.
- `ood_io.py` — Read hand-labeled OOD examples from JSONL files. Main defs/classes: `load_ood_jsonl`.
- `ood_stats.py` — Summarize the hand-labeled OOD benchmark. Main defs/classes: `summarize_ood`, `validate_ood`, `validate_complete_ood`.
- `openpii.py` — Normalize records from the OpenPII-style privacy-mask schema. Main defs/classes: `load_openpii_record`, `_parse_annotation`, `_required_text`, `_required_identifier`, `_required_annotation_text`, `_required_int`.
- `serialization.py` — Serialize normalized dataset examples as JSONL. Main defs/classes: `dataset_example_to_dict`, `dataset_example_from_dict`, `write_dataset_jsonl`, `read_dataset_jsonl`, `_character_span_from_dict`, `_require_string`, `_require_integer`, `_normalize_path`.
- `splits.py` — Deterministic sampling and splitting for normalized datasets. Main defs/classes: `DatasetSplit`, `sample_examples`, `split_examples`, `_validate_examples`, `_validate_non_negative_integer`, `_validate_seed`.
- `statistics.py` — Calculate summary statistics for normalized PII datasets. Main defs/classes: `DatasetStatistics`, `calculate_dataset_statistics`, `_validate_count_mapping`.
- `taxonomy.py` — Official source labels and their normalized PII taxonomy. Main defs/classes: `LabelAudit`, `audit_label_mapping`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
