# Evaluation contract

## Primary metric: leak rate

A gold PII span leaks when any character remains unredacted. A partial match is therefore a failure for leak rate even when token-level F1 looks acceptable.

## Always report

- leak rate;
- exact span precision, recall, and F1;
- partial-overlap span precision, recall, and F1;
- per-entity recall;
- over-redaction rate;
- expected calibration error;
- latency and model size;
- every number on in-distribution, CoNLL, and hand-labeled real text.

`research/eval/metrics.py` contains pure implementations for the currently testable metrics. Evaluation orchestration and report generation should be added before model training begins.
