# Evaluation

The evaluation package provides the span-level metric contract, baseline scoring, report I/O, and reproducible baseline evaluation used for Milestone 1.

The primary metric is leak rate. Baseline reports also include exact and partial span precision/recall/F1, per-entity recall, and over-redaction rate.

Use:

```bash
uv run python -m scripts.run_evaluation --help
uv run python -m scripts.summarize_baselines --help
```

Milestone 1 evaluates the regex and Presidio baselines on:

* the 200-example hand-labeled OOD set;
* the held-out Ai4Privacy test split;
* the CoNLL-2003 PERSON-only test benchmark.

Generated reports under `research/results/` are reproducible outputs and are not committed.
