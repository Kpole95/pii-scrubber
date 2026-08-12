# src / pii_scrub / text

Text alignment, span conversion, replacement, restoration, and window-merging primitives.

## Files

- `__init__.py` — Public text-processing primitives. Main defs/classes: No top-level defs or classes.
- `alignment.py` — Align word-level BIO labels with tokenizer subwords. Main defs/classes: `TokenEncoding`, `FastTokenizer`, `locate_words`, `validate_label_mapping`, `labels_to_ids`, `align_bio_to_subwords`, `_validate_coverage`.
- `replacement.py` — Replace detected spans with typed placeholders and restore them safely. Main defs/classes: `RestoreEntry`, `ReplacementResult`, `replace_spans`, `restore_text`, `_validate_spans`.
- `spans.py` — Convert BIO annotations and token predictions into character spans. Main defs/classes: `parse_bio_label`, `bio_tags_to_spans`, `aligned_labels_to_spans`, `token_predictions_to_spans`.
- `windows.py` — Resolve duplicate and conflicting predictions from overlapping windows. Main defs/classes: `_sort_key`, `_score`, `_overlaps`, `_validate`, `remove_exact_duplicates`, `_same_type_rank`, `resolve_same_type_overlaps`, `_cross_type_rank`, `resolve_cross_type_overlaps`, `merge_window_predictions`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
