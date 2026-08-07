"""Public text-processing primitives."""

from pii_scrub.text.alignment import (
    align_bio_to_subwords,
    labels_to_ids,
    locate_words,
    validate_label_mapping,
)
from pii_scrub.text.replacement import ReplacementResult, RestoreEntry, replace_spans, restore_text
from pii_scrub.text.spans import aligned_labels_to_spans, bio_tags_to_spans, parse_bio_label
from pii_scrub.text.windows import (
    merge_window_predictions,
    remove_exact_duplicates,
    resolve_cross_type_overlaps,
    resolve_same_type_overlaps,
)
from pii_scrub.types import AlignedExample, CharacterSpan, WindowSpan, WordOffset

__all__ = [
    "AlignedExample",
    "CharacterSpan",
    "ReplacementResult",
    "RestoreEntry",
    "WindowSpan",
    "WordOffset",
    "align_bio_to_subwords",
    "aligned_labels_to_spans",
    "bio_tags_to_spans",
    "labels_to_ids",
    "locate_words",
    "merge_window_predictions",
    "parse_bio_label",
    "remove_exact_duplicates",
    "replace_spans",
    "resolve_cross_type_overlaps",
    "resolve_same_type_overlaps",
    "restore_text",
    "validate_label_mapping",
]
