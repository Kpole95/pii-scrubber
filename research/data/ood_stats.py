"""Summarize the hand-labeled OOD benchmark."""

from collections import Counter
from collections.abc import Sequence

from research.data.models import DatasetExample


def summarize_ood(examples: Sequence[DatasetExample]) -> dict[str, object]:
    """Return compact counts for the OOD benchmark.

    Example:
        Two rows with one PERSON span return count ``2`` and
        ``{"PERSON": 1}``.
    """
    labels = Counter(span.entity_type for example in examples for span in example.spans)

    return {
        "examples": len(examples),
        "positive": sum(bool(example.spans) for example in examples),
        "negative": sum(not example.spans for example in examples),
        "entities": sum(labels.values()),
        "labels": dict(sorted(labels.items())),
    }


def validate_ood(
    examples: Sequence[DatasetExample],
    *,
    target: int = 200,
) -> None:
    """Validate benchmark-level requirements.

    Example:
        Duplicate IDs raise ``ValueError`` before evaluation starts.
    """
    ids = [example.example_id for example in examples]

    if len(ids) != len(set(ids)):
        raise ValueError("OOD example IDs must be unique")

    if len(examples) > target:
        raise ValueError(f"OOD dataset exceeds target of {target} examples")


def validate_complete_ood(
    examples: Sequence[DatasetExample],
    *,
    target: int = 200,
) -> None:
    """Require a finished OOD benchmark with exactly the target size.

    Example:
        ``200`` examples pass; ``199`` examples raise ``ValueError``.
    """
    validate_ood(examples, target=target)

    if len(examples) != target:
        raise ValueError(f"OOD dataset requires exactly {target} examples")
