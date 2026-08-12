"""Resolve duplicate and conflicting predictions from overlapping windows."""

from collections.abc import Sequence

from pii_scrub.types import WindowSpan


def _sort_key(item: WindowSpan) -> tuple[int, int, str, int]:
    """Return the stable document-order sort key."""
    return item.span.start, item.span.end, item.span.entity_type, item.window_index


def _score(item: WindowSpan) -> float:
    """Return a comparable confidence score for one prediction."""
    return item.score if item.score is not None else float("-inf")


def _overlaps(left: WindowSpan, right: WindowSpan) -> bool:
    """Return whether two spans overlap."""
    return left.span.start < right.span.end and right.span.start < left.span.end


def _validate(predictions: Sequence[WindowSpan]) -> None:
    """Validate the supplied prediction collection."""
    for item in predictions:
        if not isinstance(item, WindowSpan):
            raise TypeError("predictions must contain only WindowSpan objects")


def remove_exact_duplicates(predictions: Sequence[WindowSpan]) -> list[WindowSpan]:
    """Keep one prediction for each identical boundary-and-type tuple.

    Higher confidence wins; equal or absent scores prefer the earlier window.
    """

    _validate(predictions)
    best: dict[tuple[int, int, str], WindowSpan] = {}
    for item in predictions:
        key = item.span.start, item.span.end, item.span.entity_type
        current = best.get(key)
        if current is None or (_score(item), -item.window_index) > (
            _score(current),
            -current.window_index,
        ):
            best[key] = item
    return sorted(best.values(), key=_sort_key)


def _same_type_rank(item: WindowSpan) -> tuple[float, int, int]:
    """Return the ranking key for same-type overlap resolution."""
    return _score(item), item.span.length, -item.window_index


def resolve_same_type_overlaps(predictions: Sequence[WindowSpan]) -> list[WindowSpan]:
    """Resolve overlapping predictions of the same entity type."""

    _validate(predictions)
    resolved: list[WindowSpan] = []
    for item in sorted(predictions, key=_sort_key):
        conflict = next(
            (
                index
                for index, current in enumerate(resolved)
                if item.span.entity_type == current.span.entity_type and _overlaps(item, current)
            ),
            None,
        )
        if conflict is None:
            resolved.append(item)
        elif _same_type_rank(item) > _same_type_rank(resolved[conflict]):
            resolved[conflict] = item
    return sorted(resolved, key=_sort_key)


def _cross_type_rank(item: WindowSpan) -> tuple[float, int, int, str]:
    """Return the ranking key for cross-type overlap resolution."""
    inverse_type = "".join(chr(0x10FFFF - ord(char)) for char in item.span.entity_type)
    return _score(item), item.span.length, -item.window_index, inverse_type


def resolve_cross_type_overlaps(predictions: Sequence[WindowSpan]) -> list[WindowSpan]:
    """Resolve overlapping predictions that disagree on entity type.

    Preference order is confidence, span length, earlier window, then
    alphabetically earlier entity type.
    """

    _validate(predictions)
    resolved: list[WindowSpan] = []
    for item in sorted(predictions, key=_sort_key):
        conflict = next(
            (
                index
                for index, current in enumerate(resolved)
                if item.span.entity_type != current.span.entity_type and _overlaps(item, current)
            ),
            None,
        )
        if conflict is None:
            resolved.append(item)
        elif _cross_type_rank(item) > _cross_type_rank(resolved[conflict]):
            resolved[conflict] = item
    return sorted(resolved, key=_sort_key)


def merge_window_predictions(predictions: Sequence[WindowSpan]) -> list[WindowSpan]:
    """Return sorted, non-conflicting predictions from all windows."""

    deduplicated = remove_exact_duplicates(predictions)
    return resolve_cross_type_overlaps(resolve_same_type_overlaps(deduplicated))
