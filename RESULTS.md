# Results

This file keeps the main research numbers in one place. Leak rate is the primary privacy metric: a gold PII span counts as leaked when any part of it remains uncovered. Exact F1 requires the predicted span boundaries and entity type to match exactly. Partial F1 gives credit for overlapping spans, so it is useful for diagnosis but is not a replacement for leak rate.

## Evaluation sets

| Dataset | Size | Purpose |
|---|---:|---|
| Ai4Privacy test | 4,000 examples | Held-out in-domain evaluation |
| Hand-labeled OOD | 200 examples | More realistic text outside the training distribution |
| CoNLL-2003 | 3,453 examples | PERSON-only transfer test on news text |

Ai4Privacy validation was used for calibration and threshold selection. Final Ai4Privacy test, OOD, and CoNLL results were not used to retune thresholds.

## DeBERTa-v3-base encoder

The encoder is the selected primary model.

| Dataset | Precision | Recall | Exact F1 | Partial F1 | Leak rate | Over-redaction |
|---|---:|---:|---:|---:|---:|---:|
| Ai4Privacy | 0.9626 | 0.9460 | 0.9542 | 0.9677 | 0.0339 | 0.0007 |
| OOD 200 | 0.4226 | 0.4494 | 0.4356 | 0.6687 | 0.4873 | 0.0050 |
| CoNLL PERSON | 0.1411 | 0.2103 | 0.1689 | 0.6473 | 0.7897 | 0.0050 |

Exact-count details:

| Dataset | TP | FP | FN |
|---|---:|---:|---:|
| Ai4Privacy | 6,744 | 262 | 385 |
| OOD 200 | 142 | 194 | 174 |
| CoNLL PERSON | 340 | 2,069 | 1,277 |

The CoNLL table uses the final PERSON-only prediction policy. An older report allowed non-PERSON predictions into the score and is not used here.

## Qwen2.5-1.5B + LoRA

Qwen was trained to return JSON spans rather than rewrite the input text. This kept the task measurable, but the model still had exact-offset and parse-reliability problems.

| Dataset | Exact F1 | Partial F1 | Leak rate | Over-redaction | Parse failures |
|---|---:|---:|---:|---:|---:|
| Ai4Privacy | 0.4658 | 0.9321 | 0.5246 | 0.0113 | 4.68% |
| OOD 200 | 0.1498 | 0.5957 | 0.8070 | 0.0224 | 8.00% |
| CoNLL PERSON | 0.0955 | 0.6028 | 0.8670 | 0.0187 | 9.30% |

The large gap between exact and partial F1 is important. Qwen often identifies the right semantic region but does not produce boundaries reliably enough for a privacy scrubber. The DeBERTa encoder is therefore the main model family for Phase 1.

## Calibration

The encoder's global expected calibration error on Ai4Privacy validation was:

```text
ECE = 0.012570723746939225
```

Calibration quality varied by entity. Threshold profiles were selected on validation only and then frozen.

## Frozen threshold profiles

### Balanced

| Entity | Threshold | Entity | Threshold |
|---|---:|---|---:|
| ADDRESS | 0.55 | BANK_ACCOUNT | 0.80 |
| COMPANY | 0.90 | CREDIT_CARD | 0.75 |
| DATE | 0.80 | EMAIL | 0.95 |
| ID_NUMBER | 0.60 | IP_ADDRESS | 0.95 |
| LOCATION | 0.95 | PASSWORD | 0.90 |
| PERSON | 0.60 | PHONE | 0.95 |
| POSTAL_CODE | 0.50 | SECRET | 0.60 |
| SOCIAL_SECURITY_NUMBER | 0.65 | SWIFT_CODE | 0.85 |
| TITLE | 0.90 | USERNAME | 0.60 |

### Strict

| Entity | Threshold | Entity | Threshold |
|---|---:|---|---:|
| ADDRESS | 0.40 | BANK_ACCOUNT | 0.40 |
| COMPANY | 0.55 | CREDIT_CARD | 0.50 |
| DATE | 0.60 | EMAIL | 0.95 |
| ID_NUMBER | 0.50 | IP_ADDRESS | 0.90 |
| LOCATION | 0.95 | PASSWORD | 0.90 |
| PERSON | 0.50 | PHONE | 0.35 |
| POSTAL_CODE | 0.40 | SECRET | 0.45 |
| SOCIAL_SECURITY_NUMBER | 0.40 | SWIFT_CODE | 0.70 |
| TITLE | 0.75 | USERNAME | 0.60 |

The frozen profiles are policy choices, not claims that thresholding improved every final benchmark. In the final evaluation, the raw encoder had the lowest observed leak rate on Ai4Privacy, OOD, and CoNLL.

## Stage 13 robustness audit

The robustness audit used the raw encoder as the fixed reference model. Thresholds were not retuned.

### Ai4Privacy robustness subset

Baseline on the 200-example subset:

| Metric | Value |
|---|---:|
| Exact F1 | 0.9541 |
| Partial F1 | 0.9748 |
| Leak rate | 0.0523 |
| Over-redaction | 0.0005 |

Perturbation deltas are measured against the same examples before the change.

| Change | Changed examples | Leak delta | Exact F1 delta | Partial F1 delta |
|---|---:|---:|---:|---:|
| lowercase | 200 | +0.2733 | -0.2538 | -0.1262 |
| uppercase | 200 | +0.1134 | -0.1471 | -0.0750 |
| double spaces | 200 | +0.0000 | +0.0000 | +0.0000 |
| non-breaking spaces | 200 | +0.0000 | +0.0000 | +0.0000 |
| Unicode dashes | 66 | +0.0082 | -0.0044 | -0.0044 |
| curly apostrophes | 52 | -0.0099 | +0.0102 | +0.0000 |

The largest weakness is casing, especially full lowercasing.

### OOD robustness

OOD baseline:

| Metric | Value |
|---|---:|
| Exact F1 | 0.4356 |
| Partial F1 | 0.6687 |
| Leak rate | 0.4873 |
| Over-redaction | 0.0050 |

| Change | Changed examples | Leak delta | Exact F1 delta | Partial F1 delta |
|---|---:|---:|---:|---:|
| lowercase | 185 | +0.0447 | -0.0915 | -0.0675 |
| uppercase | 200 | +0.0032 | -0.0416 | -0.0389 |
| double spaces | 199 | +0.0000 | +0.0000 | +0.0000 |
| non-breaking spaces | 199 | +0.0000 | +0.0000 | +0.0000 |
| Unicode dashes | 48 | +0.0256 | -0.0456 | -0.0154 |
| curly apostrophes | 6 | +0.1250 | -0.1238 | -0.0952 |

The curly-apostrophe result comes from only six changed examples, so it is kept as a warning rather than treated as a broad conclusion.

### Long-context stress

Seventeen labeled OOD examples were tested after adding long prefixes.

| Variant | Leak delta | Exact F1 delta | Partial F1 delta |
|---|---:|---:|---:|
| 480-word prefix | -0.0741 | +0.0328 | -0.0328 |
| 560-word prefix | -0.0370 | -0.0125 | -0.0239 |

Leak rate did not increase in either long-context test. This supports the windowing and merge path, but the sample is small and should not be read as a universal guarantee.

## Memorization audit

The Stage 13 screen found:

- zero exact full-text duplicates from training into validation, test, or OOD;
- zero normalized full-text duplicates after case folding and whitespace collapse;
- very little PII-value reuse in the OOD set;
- high PERSON-value reuse inside Ai4Privacy.

A targeted PERSON check found:

| PERSON group | Gold spans | Exact TP | Exact recall |
|---|---:|---:|---:|
| value seen in training | 1,904 | 1,818 | 0.9548 |
| value not seen in training | 49 | 43 | 0.8776 |

The seen-minus-unseen recall gap is **7.73 percentage points**. This is a useful warning about value reuse, but the unseen group is small and there was no full-example train/test duplication. It is therefore documented as sensitivity to repeated values, not proof of memorized examples.

## What the results mean

1. DeBERTa is the right primary architecture for this task in Phase 1.
2. In-domain performance is strong, but real/OOD performance is much weaker.
3. Partial F1 can hide boundary failures, especially for a generative model.
4. Calibration is good overall, but frozen thresholds do not automatically improve privacy on every final dataset.
5. Casing is the clearest robustness weakness found in Stage 13.
6. Redaction remains risk reduction, not a guarantee of anonymization or perfect recall.
