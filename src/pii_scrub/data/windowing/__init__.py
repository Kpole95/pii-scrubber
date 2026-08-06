"""Public API for long-document windowing utilities."""

from .merging import (
    merge_window_predictions,
    remove_exact_duplicates,
    resolve_cross_type_overlaps,
    resolve_same_type_overlaps,
)
from .models import WindowSpan

__all__ = [
    "WindowSpan",
    "merge_window_predictions",
    "remove_exact_duplicates",
    "resolve_cross_type_overlaps",
    "resolve_same_type_overlaps",
]