# Development

## Local setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
python scripts/check_structure.py
```

## Module standard

Public modules, classes, and functions require purpose-focused docstrings. Examples belong on APIs whose behaviour is not obvious. Private helpers use short docstrings only when they encode a non-obvious rule.

Use guard clauses, explicit types, immutable data models, and functions with one responsibility. Comments explain why a decision exists rather than narrating the code.

## Adding a model

1. Implement the runtime detector adapter behind `Detector`.
2. Add contract tests using injected fakes.
3. Add the training entry point under `research/train`.
4. Commit a reproducible config and seed.
5. Evaluate against all three required test sets.
6. Add model and dataset card updates.
