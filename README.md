# PII Scrubber

PII Scrubber is a local Python library and CLI for finding and replacing personal data before text is sent somewhere else.

The repository also contains the research code used to compare rules, Microsoft Presidio, a DeBERTa encoder, and a Qwen LoRA model using the same span-level metrics.

The main research result is more specific: **DeBERTa is the selected primary learned model and leads the Ai4Privacy and hand-labeled OOD comparisons**. Presidio remains stronger on the canonical CoNLL PERSON benchmark, where annotation conventions differ substantially from Ai4Privacy.

## What it does

- detects PII with local detectors;
- replaces detected spans with typed placeholders;
- keeps a restoration map so text can be restored by the caller;
- preserves document-level character offsets through tokenization and windows;
- supports a regex runtime default plus Presidio, DeBERTa encoder, and generative detector adapters; DeBERTa is the primary learned model, while Qwen is a research comparison;
- evaluates leak rate, exact span F1, partial span F1, per-entity recall, and over-redaction;
- includes a 200-example hand-labeled OOD benchmark under [`research/labeled_ood/`](research/labeled_ood/);
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

> **Important:** the numbers below are from the separately trained DeBERTa research model. Installing the package does **not** download or enable these weights. `Scrubber()` still uses the regex detector by default.

The trained encoder artifact is kept outside normal Git history. From a source checkout with the ML dependencies installed and a local copy of the trained model, the encoder can be loaded like this:

```python
# This script lives outside src/pii_scrub; the runtime package does not import research.
from pathlib import Path

from pii_scrub import Scrubber
from pii_scrub.detectors.encoder import EncoderDetector
from research.eval.encoder import load_encoder_predictor

predictor = load_encoder_predictor(Path("/path/to/encoder-model"))
scrubber = Scrubber(detector=EncoderDetector(predictor))

result = scrubber.scrub("Contact Ana at ana@example.com")
print(result.text)
```

`load_encoder_predictor(...)` is the raw model path used by the research evaluation. Wrapping it in `Scrubber` applies the runtime threshold policy. The model-loading helper lives in the repository research code; the published runtime package does not bundle the trained weights.

The table below reports the **raw DeBERTa encoder**, before calibrated runtime profile filtering:

| Dataset | Exact F1 | Partial F1 | Leak rate | Over-redaction |
|---|---:|---:|---:|---:|
| Ai4Privacy test | 0.9542 | 0.9677 | 0.0339 | 0.0007 |
| OOD 200 | 0.4356 | 0.6687 | 0.4873 | 0.0050 |
| CoNLL PERSON | 0.1689 | 0.6473 | 0.7897 | 0.0050 |

DeBERTa is the primary learned model because it is decisively strongest on Ai4Privacy, edges Presidio on the hand-labeled OOD benchmark, and is much stronger than Qwen on exact spans and leak rate.

CoNLL PERSON is the important exception. Presidio reached **0.7798 exact F1** with a **0.2257 leak rate**, versus **0.1689 exact F1** and **0.7897 leak** for raw DeBERTa. A research-only PERSON-fragment merge raised DeBERTa to **0.7429 exact F1** and **0.2597 leak**, showing that much of the raw CoNLL gap comes from different name-span annotation conventions. That merge is not the production default because it hurts Ai4Privacy exact-span behavior.

The Qwen2.5-1.5B LoRA model was much weaker on exact spans and leak rate. Its partial-span scores show that it often identifies the correct area of the text, but its exact offsets and structured output were not reliable enough for the main redaction path.

See `RESULTS.md` for the full model comparison, calibration results, frozen thresholds, robustness checks, and memorization audit.

## Robustness findings

The robustness evaluation found one important weakness: **case changes can hurt the encoder**.

- lowercasing the Ai4Privacy robustness subset increased leak rate by 27.33 percentage points;
- uppercasing increased leak rate by 11.34 percentage points;
- double spaces and non-breaking spaces caused essentially no leak-rate change;
- 480-word and 560-word prefix stress did not increase OOD leak rate;
- the OOD curly-apostrophe perturbation increased leak by 12.5 percentage points, but only six examples changed, so this is a warning rather than a broad conclusion;
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
