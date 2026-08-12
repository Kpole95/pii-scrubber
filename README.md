# PII Scrubber

PII Scrubber is a local Python library and CLI for finding and replacing personal data before text is sent somewhere else.

The repository also contains the research code used to compare rules, Microsoft Presidio, a DeBERTa encoder, and a Qwen LoRA model using the same span-level metrics.

The main research result is simple: **the DeBERTa encoder is the strongest model in this study**. It gives a much lower leak rate and much better exact span accuracy than the Qwen model on the frozen evaluation sets.

## What it does

- detects PII with local detectors;
- replaces detected spans with typed placeholders;
- keeps a restoration map so text can be restored by the caller;
- preserves document-level character offsets through tokenization and windows;
- supports regex, Presidio, encoder, and generative detector adapters;
- evaluates leak rate, exact span F1, partial span F1, per-entity recall, and over-redaction;
- includes a 200-example hand-labeled OOD benchmark;
- includes calibration, threshold-profile, robustness, and memorization audits.

The base runtime has no required ML dependency.

`Scrubber()` uses the regex detector by default. The default detector currently handles email addresses, phone numbers, and IPv4 addresses. Broader detection can be provided through optional detectors or an injected model predictor.

## Installation

Install from source:

```bash
python -m pip install .
```

Optional dependency groups:

```bash
python -m pip install ".[presidio]"
python -m pip install ".[ml]"
python -m pip install ".[research,dev]"
```

The package can also be built as a wheel and source distribution:

```bash
python -m build
```

Model weights are intentionally not stored in normal Git history.

## Quick start

### Python

```python
from pii_scrub import Scrubber

text = "Email ana@example.com or call +44 7700 900123."

result = Scrubber().scrub(text)

print(result.text)
# Email [EMAIL_1] or call [PHONE_1].

original = Scrubber.restore(result.text, result.mapping)

assert original == text
```

### CLI

```bash
pii-scrub "Email ana@example.com"
# Email [EMAIL_1]
```

You can also read text from a file and write the restoration mapping:

```bash
pii-scrub \
  --file transcript.txt \
  --entities EMAIL,PHONE \
  --mapping-out mapping.json
```

## Project structure

```text
configs/               training and threshold settings
src/pii_scrub/         installable runtime package
research/data/         dataset normalization and artifacts
research/eval/         metrics, calibration, and model evaluation
research/train/        DeBERTa and Qwen training code
research/labeled_ood/  200 reviewed OOD examples
scripts/               experiment and repository commands
tests/                 unit, contract, and integration tests
```

Each main folder has a short `README.md` that explains its files and important definitions.

The runtime package never imports `research`. Research code may use the runtime package, but not the other way around.

## Main results

The raw DeBERTa encoder is the reference model because it had the lowest final leak rate on all three frozen benchmarks.

| Dataset | Exact F1 | Partial F1 | Leak rate | Over-redaction |
|---|---:|---:|---:|---:|
| Ai4Privacy test | 0.9542 | 0.9677 | 0.0339 | 0.0007 |
| OOD 200 | 0.4356 | 0.6687 | 0.4873 | 0.0050 |
| CoNLL PERSON | 0.1689 | 0.6473 | 0.7897 | 0.0050 |

The Qwen2.5-1.5B LoRA model was much weaker on exact spans and leak rate.

Its partial-span scores show that it often identifies the correct area of the text, but its exact offsets and structured output were not reliable enough for the main redaction path.

See `RESULTS.md` for the full model comparison, calibration results, frozen thresholds, robustness checks, and memorization audit.

## Robustness findings

The robustness evaluation found one important weakness: **case changes can hurt the encoder**.

- lowercasing the Ai4Privacy robustness subset increased leak rate by 27.33 percentage points;
- uppercasing increased leak rate by 11.34 percentage points;
- double spaces and non-breaking spaces caused essentially no leak-rate change;
- 480-word and 560-word prefix stress did not increase OOD leak rate;
- no exact or normalized full-text train duplicates were found in validation, test, or OOD data.

These checks were measured without retraining or retuning thresholds on the final test sets.

## Known limitations

- Redaction is **not anonymization**. Remaining context can still identify someone.
- Recall is not 100%. Treat the tool as risk reduction, not a privacy guarantee.
- OOD and CoNLL performance is much weaker than in-domain Ai4Privacy performance.
- The encoder is sensitive to casing, especially fully lowercased text.
- The training corpus is more templated than many real-world documents.
- Scanned PDFs, OCR, DOCX parsing, streaming, a web UI, and cloud fallback are not supported.
- The default regex detector covers only a small structured subset of the full research taxonomy.
- Model weights are not distributed through the Git repository.

## Development checks

```bash
python -m pytest
python scripts/check_structure.py
ruff format --check .
ruff check .
mypy src scripts research
```

The structure check keeps project Python modules at 300 lines or less, checks required docstrings, and prevents runtime imports from the research package.

## Release

The current public release is:

```text
v0.1.0
```

The release includes:

- the Python runtime package;
- CLI support;
- reversible redaction;
- configurable threshold profiles;
- research and evaluation code;
- 468 automated tests;
- wheel and source-distribution builds;
- clean-install CLI and Python API smoke tests.

## License

PII Scrubber is released under the MIT License.

See `LICENSE` for details.
