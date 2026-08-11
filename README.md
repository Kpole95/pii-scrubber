# PII Scrubber

PII Scrubber is a local-first Python library and CLI that detects and replaces personal data before text is sent to another system. The repository also contains the research pipeline used to compare rule-based detection, Presidio, an encoder classifier, and a generative small language model at span level.

## Current status

The repository currently contains the completed **Milestone 1 foundation through Stage 12**:

- strict BIO and subword alignment with document-global offsets;
- exact character-span reconstruction and overlapping-window merging;
- normalized Ai4Privacy and CoNLL-2003 benchmark adapters;
- deterministic splits, JSONL artifacts, manifests, and statistics;
- a fixed 200-example hand-labeled OOD benchmark;
- reversible replacement plus regex, Presidio, and encoder detector support;
- leak rate, exact/partial span F1, per-entity recall, over-redaction, and calibration metrics;
- trained DeBERTa-v3-base encoder evaluation across Ai4Privacy, OOD, and CoNLL;
- trained Qwen2.5-1.5B LoRA evaluation with strict JSON span parsing;
- validation-only encoder calibration with frozen `balanced` and `strict` threshold profiles;
- runtime `recall_mode` support for the frozen profiles;
- 464 offline tests.

The encoder is the selected primary model family. It has substantially lower leak rate and higher exact-span F1 than the Qwen model across all three frozen benchmarks. Qwen shows useful semantic recognition through partial-span scores, but exact offsets and structured-output reliability are not strong enough for the scrubber's primary detection path.

Threshold profiles were selected only on the Ai4Privacy validation split. Final Ai4Privacy test, OOD, and CoNLL measurements are reported without test-set retuning. The raw encoder currently has the lowest observed leak rate on all three final benchmarks, while the frozen profiles remain available as explicit runtime policy choices.

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

The default detector is the regex baseline. The trained encoder and Presidio integrations provide broader detection for names and contextual entities.

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
│   ├── train/                  encoder and Qwen training pipelines
│   ├── labeled_ood/            hand-labeled real-text test set
│   └── results/                reproducible generated evaluation reports
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
