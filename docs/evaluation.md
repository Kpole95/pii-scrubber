# Evaluation contract

## Primary metric: leak rate

A gold PII span leaks when any character remains unredacted. A partial
match is therefore a failure for leak rate even when span-level F1 looks
acceptable.

## Core metrics

Baseline evaluation reports:

- leak rate;
- exact span precision, recall, and F1;
- partial-overlap span precision, recall, and F1;
- per-entity recall;
- over-redaction rate.

Expected calibration error, latency, and model size become relevant when
learned models are evaluated.

## Required benchmarks

Milestone 1 uses three evaluation sets:

- held-out Ai4Privacy;
- CoNLL-2003 as a PERSON-only out-of-distribution benchmark;
- 200 hand-labeled real/OOD examples.

Both regex and Microsoft Presidio are evaluated on all three datasets.

## Reproducible evaluation

`scripts/run_evaluation.py` runs one baseline evaluation and writes a
JSON report.

`scripts/summarize_baselines.py` combines baseline JSON reports into a
Markdown comparison table.

Generated outputs under `research/results/` are ignored by Git and
should be reproduced from the committed evaluation code.