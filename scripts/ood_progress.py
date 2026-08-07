"""Print progress for the hand-labeled OOD benchmark."""

from pathlib import Path

from research.data.ood_io import load_ood_jsonl
from research.data.ood_stats import summarize_ood, validate_ood

DATASET = Path("research/labeled_ood/examples.jsonl")


def main() -> None:
    """Print OOD labeling progress.

    Example:
        ``5/200 examples`` with positive, negative, and label counts.
    """
    examples = load_ood_jsonl(DATASET)
    validate_ood(examples)
    stats = summarize_ood(examples)

    print(f"{stats['examples']}/200 examples")
    print(f"positive: {stats['positive']}")
    print(f"negative: {stats['negative']}")
    print(f"entities: {stats['entities']}")
    print(f"labels: {stats['labels']}")


if __name__ == "__main__":
    main()
