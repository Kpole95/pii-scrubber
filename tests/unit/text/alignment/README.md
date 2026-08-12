# tests / unit / text / alignment

Tests for BIO labels, tokenizer alignment, word offsets, and span reconstruction.

## Files

- `__init__.py` — Text alignment unit tests. Main defs/classes: No top-level defs or classes.
- `test_bio.py` — Tests for converting word BIO labels to character spans. Main defs/classes: `test_bio_tags_to_spans_builds_multiword_entity`, `test_bio_tags_to_spans_builds_multiple_entities`, `test_bio_tags_to_spans_closes_entity_at_end`, `test_bio_tags_to_spans_handles_adjacent_entities`, `test_bio_tags_to_spans_returns_empty_list_for_all_o`, `test_bio_tags_to_spans_rejects_different_lengths`, plus 2 more tests.
- `test_predictions.py` — Tests for converting model BIO predictions into detected spans. Main defs/classes: `test_token_predictions_build_scored_entity`, `test_token_predictions_support_overlapping_sentencepiece_offsets`, `test_token_predictions_repair_orphan_i`, `test_token_predictions_repair_i_type_change`, `test_token_predictions_ignore_zero_length_special_tokens`, `test_token_predictions_reject_mismatched_lengths`, plus 1 more test.
- `test_reconstruction.py` — Tests for reconstructing spans from token BIO labels. Main defs/classes: `test_reconstruction_builds_multiple_entities`, `test_reconstruction_returns_empty_for_all_o`, `test_reconstruction_rejects_orphan_i`, `test_reconstruction_rejects_type_change`, `test_ignored_continuation_extends_entity`, `test_special_token_does_not_extend_entity`.
- `test_round_trip.py` — End-to-end round-trip tests for the alignment package. Main defs/classes: `_assert_round_trip`, `test_multiword_entity_round_trips`, `test_split_single_word_entity_round_trips`, `test_multiple_entities_round_trip`, `test_irregular_whitespace_round_trips`, `test_unicode_entity_round_trips`.
- `test_subwords.py` — Tests for word BIO to tokenizer-subword alignment. Main defs/classes: `test_all_subwords_propagates_labels`, `test_first_subword_ignores_later_pieces`, `test_alignment_returns_global_text_offsets`, `test_alignment_preserves_word_ids`, `test_special_tokens_have_no_labels`, `test_alignment_rejects_unknown_strategy`, plus 8 more tests.
- `test_words.py` — Tests for locating source words. Main defs/classes: `test_locate_words_maps_exact_offsets`, `test_locate_words_handles_repeated_words`, `test_locate_words_preserves_irregular_whitespace`, `test_locate_words_handles_unicode`, `test_locate_words_returns_empty_list`, `test_locate_words_rejects_missing_word`, plus 1 more test.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
