# tests / unit / text / windows

Tests for overlapping-window boundaries, deduplication, merging, and conflict resolution.

## Files

- `__init__.py` — Text windows unit tests. Main defs/classes: No top-level defs or classes.
- `test_merging.py` — Tests for the public window-prediction merge pipeline. Main defs/classes: `test_merge_window_predictions_removes_duplicates_and_overlaps`, `test_merge_window_predictions_resolves_different_type_overlap`, `test_merge_window_predictions_returns_empty_list`, `test_merge_window_predictions_returns_sorted_output`, `test_merge_window_predictions_resolves_cross_type_conflict`.
- `test_window_boundaries.py` — Integration tests for entities predicted near window boundaries. Main defs/classes: `test_same_entity_from_two_windows_becomes_one_span`, `test_boundary_disagreement_keeps_best_span`, `test_entity_crossing_window_boundary_is_preserved`, `test_separate_entities_from_neighboring_windows_are_preserved`, `test_three_windows_repeating_one_entity_produce_one_result`, `test_boundary_type_conflict_keeps_best_prediction`, plus 2 more tests.
- `test_window_duplicates.py` — Tests for removing duplicate predictions from overlapping windows. Main defs/classes: `test_remove_exact_duplicates_keeps_one_prediction`, `test_remove_exact_duplicates_keeps_highest_score`, `test_remove_exact_duplicates_keeps_earlier_window_when_scores_equal`, `test_remove_exact_duplicates_treats_missing_score_as_lower`, `test_remove_exact_duplicates_keeps_different_entity_types`, `test_remove_exact_duplicates_keeps_different_boundaries`, plus 3 more tests.
- `test_window_overlap_resolution.py` — Tests for resolving overlapping predictions from neighboring windows. Main defs/classes: `test_resolve_same_type_overlaps_keeps_highest_score`, `test_resolve_same_type_overlaps_keeps_longer_span_when_scores_equal`, `test_resolve_same_type_overlaps_keeps_earlier_window_on_full_tie`, `test_resolve_same_type_overlaps_keeps_non_overlapping_spans`, `test_resolve_same_type_overlaps_keeps_different_types`, `test_resolve_same_type_overlaps_treats_touching_spans_as_separate`, plus 5 more tests.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
