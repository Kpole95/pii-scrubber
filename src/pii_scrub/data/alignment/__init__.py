"""Public API for BIO, word, and tokenizer alignment."""

from .bio import bio_tags_to_spans
from .models import AlignedExample, CharacterSpan, WordOffset
from .reconstruction import aligned_labels_to_spans
from .subwords import align_bio_to_subwords
from .types import SubwordLabelStrategy
from .validation import (
    labels_to_ids,
    parse_bio_label,
    validate_label_mapping,
)
from .words import locate_words

__all__ = [
    "AlignedExample",
    "CharacterSpan",
    "SubwordLabelStrategy",
    "WordOffset",
    "align_bio_to_subwords",
    "aligned_labels_to_spans",
    "bio_tags_to_spans",
    "labels_to_ids",
    "locate_words",
    "parse_bio_label",
    "validate_label_mapping",
]