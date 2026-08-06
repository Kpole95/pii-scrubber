"""Merge entity predictions produced by overlapping windows."""

from collections.abc import Sequence

from .models import WindowSpan


def remove_exact_duplicates(
    predictions: Sequence[WindowSpan],
) -> list[WindowSpan]:
    """Remove predictions with identical span boundaries and entity types.

    When duplicates have confidence scores, the highest-scoring prediction
    is kept. If scores are missing or equal, the earlier window is kept.

    Example:
        Two windows predict ``PERSON`` at characters 100–112.
        The returned list contains only one of them.
    """

    best_by_span: dict[tuple[int, int, str], WindowSpan] = {}

    for prediction in predictions:
        if not isinstance(prediction, WindowSpan):
            raise TypeError(
                "predictions must contain only WindowSpan objects"
            )

        key = (
            prediction.span.start,
            prediction.span.end,
            prediction.span.entity_type,
        )

        current = best_by_span.get(key)

        if current is None:
            best_by_span[key] = prediction
            continue

        current_score = (
            current.score
            if current.score is not None
            else float("-inf")
        )
        new_score = (
            prediction.score
            if prediction.score is not None
            else float("-inf")
        )

        if new_score > current_score:
            best_by_span[key] = prediction
            continue

        if (
            new_score == current_score
            and prediction.window_index < current.window_index
        ):
            best_by_span[key] = prediction

    return sorted(
        best_by_span.values(),
        key=lambda item: (
            item.span.start,
            item.span.end,
            item.span.entity_type,
            item.window_index,
        ),
    )

def _is_better_prediction(
    *,
    candidate: WindowSpan,
    current: WindowSpan,
) -> bool:
    """Return whether the candidate should replace the current prediction."""

    candidate_score = (
        candidate.score
        if candidate.score is not None
        else float("-inf")
    )
    current_score = (
        current.score
        if current.score is not None
        else float("-inf")
    )

    if candidate_score != current_score:
        return candidate_score > current_score

    candidate_length = candidate.span.length
    current_length = current.span.length

    if candidate_length != current_length:
        return candidate_length > current_length

    return candidate.window_index < current.window_index

def resolve_same_type_overlaps(
    predictions: Sequence[WindowSpan],
) -> list[WindowSpan]:
    """Resolve overlapping predictions with the same entity type.

    The preferred prediction is selected by:

    1. Higher confidence score.
    2. Longer character span.
    3. Earlier window index.

    Predictions with different entity types are kept because they represent
    a separate conflict that needs a later policy.

    Example:
        PERSON (100, 112), score 0.80
        PERSON (100, 113), score 0.95

        The second prediction is kept.
    """

    for prediction in predictions:
        if not isinstance(prediction, WindowSpan):
            raise TypeError(
                "predictions must contain only WindowSpan objects"
            )

    sorted_predictions = sorted(
        predictions,
        key=lambda item: (
            item.span.start,
            item.span.end,
            item.span.entity_type,
            item.window_index,
        ),
    )

    resolved: list[WindowSpan] = []

    for prediction in sorted_predictions:
        conflict_index: int | None = None

        for index, current in enumerate(resolved):
            same_type = (
                prediction.span.entity_type
                == current.span.entity_type
            )

            overlaps = (
                prediction.span.start < current.span.end
                and current.span.start < prediction.span.end
            )

            if same_type and overlaps:
                conflict_index = index
                break

        if conflict_index is None:
            resolved.append(prediction)
            continue

        current = resolved[conflict_index]

        if _is_better_prediction(
            candidate=prediction,
            current=current,
        ):
            resolved[conflict_index] = prediction

    return sorted(
        resolved,
        key=lambda item: (
            item.span.start,
            item.span.end,
            item.span.entity_type,
            item.window_index,
        ),
    )

def merge_window_predictions(
    predictions: Sequence[WindowSpan],
) -> list[WindowSpan]:
    """Merge predictions produced by overlapping document windows.

    The merge happens in two stages:

    1. Remove exact duplicate spans.
    2. Resolve overlapping spans with the same entity type.

    Different entity types are preserved for a later conflict policy.

    Example:
        Two windows predict the same PERSON span, while another predicts
        a slightly longer PERSON span with higher confidence.

        The result contains only the best PERSON prediction.
    """

    deduplicated = remove_exact_duplicates(predictions)

    same_type_resolved = resolve_same_type_overlaps(
        deduplicated
    )

    return resolve_cross_type_overlaps(
        same_type_resolved
    )

def _is_better_cross_type_prediction(
    *,
    candidate: WindowSpan,
    current: WindowSpan,
) -> bool:
    """Return whether a cross-type candidate should replace the current one."""

    candidate_score = (
        candidate.score
        if candidate.score is not None
        else float("-inf")
    )
    current_score = (
        current.score
        if current.score is not None
        else float("-inf")
    )

    if candidate_score != current_score:
        return candidate_score > current_score

    candidate_length = candidate.span.length
    current_length = current.span.length

    if candidate_length != current_length:
        return candidate_length > current_length

    if candidate.window_index != current.window_index:
        return candidate.window_index < current.window_index

    return (
        candidate.span.entity_type
        < current.span.entity_type
    )

def resolve_cross_type_overlaps(
    predictions: Sequence[WindowSpan],
) -> list[WindowSpan]:
    """Resolve overlapping predictions with different entity types.

    The preferred prediction is selected by:

    1. Higher confidence score.
    2. Longer character span.
    3. Earlier window index.
    4. Alphabetically earlier entity type.

    Example:
        PERSON   (100, 112), score 0.80
        LOCATION (100, 112), score 0.95

        The LOCATION prediction is kept.
    """

    for prediction in predictions:
        if not isinstance(prediction, WindowSpan):
            raise TypeError(
                "predictions must contain only WindowSpan objects"
            )

    sorted_predictions = sorted(
        predictions,
        key=lambda item: (
            item.span.start,
            item.span.end,
            item.span.entity_type,
            item.window_index,
        ),
    )

    resolved: list[WindowSpan] = []

    for prediction in sorted_predictions:
        conflict_index: int | None = None

        for index, current in enumerate(resolved):
            different_type = (
                prediction.span.entity_type
                != current.span.entity_type
            )

            overlaps = (
                prediction.span.start < current.span.end
                and current.span.start < prediction.span.end
            )

            if different_type and overlaps:
                conflict_index = index
                break

        if conflict_index is None:
            resolved.append(prediction)
            continue

        current = resolved[conflict_index]

        if _is_better_cross_type_prediction(
            candidate=prediction,
            current=current,
        ):
            resolved[conflict_index] = prediction

    return sorted(
        resolved,
        key=lambda item: (
            item.span.start,
            item.span.end,
            item.span.entity_type,
            item.window_index,
        ),
    )