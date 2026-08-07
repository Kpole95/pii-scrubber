"""Read hand-labeled OOD examples from JSONL files."""

import json
from pathlib import Path

from research.data.models import DatasetExample
from research.data.ood import load_ood_record


def load_ood_jsonl(path: str | Path) -> tuple[DatasetExample, ...]:
    """Load one JSON object per line from an OOD JSONL file.

    Example:
        ``load_ood_jsonl("research/labeled_ood/examples.jsonl")``
        returns normalized ``DatasetExample`` objects.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    examples = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
            examples.append(load_ood_record(record))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error

    return tuple(examples)
