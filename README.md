# PII Scrubber

PII Scrubber is a local-first Python library and CLI that detects and replaces personal data before text is sent to another system. The repository also contains the research pipeline used to compare rule-based detection, Presidio, an encoder classifier, and a generative small language model at span level.

## Current status

This archive contains a cleaned and refactored **Milestone 1 foundation**:

- strict BIO and subword alignment with document-global offsets;
- exact character-span reconstruction and overlapping-window merging;
- OpenPII and Gretel Finance normalization;
- deterministic splits, JSONL artifacts, manifests, and statistics;
- reversible replacement plus a functional regex-backed API and CLI;
- leak rate, exact/partial span F1, per-entity recall, over-redaction, and ECE;
- 282 offline tests.

Encoder and generative training loops are not implemented in this ZIP. Empty model-training files would create false progress, so those modules should be added only with executable behaviour and tests.

## Install

```bash
python -m pip install .
```

Optional dependency groups:

```bash
python -m pip install ".[presidio]"
python -m pip install ".[ml]"
python -m pip install ".[research,dev]"
```

## Library usage

```python
from pii_scrub import Scrubber

text = "Email ana@example.com or call +44 7700 900123."
result = Scrubber().scrub(text)

print(result.text)
# Email [EMAIL_1] or call [PHONE_1].

assert Scrubber.restore(result.text, result.mapping) == text
```

The default detector is the regex baseline. Production use with names and contextual entities requires the later encoder or Presidio detector.

## CLI usage

```bash
pii-scrub "Email ana@example.com"
# Email [EMAIL_1]

pii-scrub --file transcript.txt --entities EMAIL,PHONE --mapping-out mapping.json
```

## Canonical structure

```text
pii-scrubber/
├── configs/                    reproducible model and decision configs
├── src/pii_scrub/              installable runtime package
│   ├── api.py                  public Scrubber orchestration
│   ├── calibration.py          runtime threshold filtering
│   ├── config.py               immutable runtime configuration
│   ├── detectors/              regex, Presidio, encoder, generative adapters
│   ├── text/                   alignment, spans, windows, replacement
│   └── types.py                shared immutable domain models
├── research/                   never imported by the runtime package
│   ├── data/                   dataset normalization, splits, artifacts
│   ├── eval/                   span-level metrics
│   ├── train/                  added when training loops are implemented
│   ├── labeled_ood/            hand-labeled real-text test set
│   └── results/                committed JSON and Markdown reports
├── scripts/                    repository and experiment entry points
├── tests/
│   ├── unit/
│   ├── contract/
│   └── integration/
└── docs/
```

The dependency rule is one-way:

```text
research -> pii_scrub
pii_scrub -X-> research
```

## Quality commands

```bash
python -m pytest
python scripts/check_structure.py
ruff format --check .
ruff check .
mypy
```

Tests use a deterministic tokenizer fake. They do not download a model or require network access.

## Non-goals

This project does not include a web UI, hosted service, cloud-model fallback, streaming pipeline, PDF/DOCX parsing, custom annotation application, or quasi-identifier anonymization.

## Safety limitation

Redaction is not anonymization. Context that remains after direct identifiers are removed may still identify a person. Recall is also never guaranteed to be 100%; report measured leak rate instead of claiming complete protection.

## Public-release blocker

A project license has not been selected. Do not publish the repository as reusable open source until an appropriate `LICENSE` file is deliberately chosen.
