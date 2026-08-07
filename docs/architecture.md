# Architecture

## Boundaries

`src/pii_scrub` is the installed product. It may use only the Python standard library unless an optional detector is explicitly instantiated. `research` contains dataset, training, evaluation, and reporting code and may import the runtime package. The reverse import is forbidden.

## Runtime flow

```text
text
  -> detector returns document-global DetectedSpan objects
  -> calibrated per-entity thresholds
  -> validate and sort spans
  -> replace right-to-left with typed placeholders
  -> redacted text plus caller-owned restore mapping
```

## Data flow

```text
raw record
  -> source-specific schema validation
  -> canonical entity taxonomy
  -> immutable DatasetExample
  -> deterministic split
  -> JSONL files plus versioned manifest
```

## Correctness-critical modules

- `text/alignment.py`: projects word labels and tokenizer-local offsets back into source text.
- `text/spans.py`: reconstructs entities from BIO predictions.
- `text/windows.py`: resolves duplicate and conflicting predictions from overlap windows.
- `text/replacement.py`: preserves offset validity and exact restoration.
- `research/eval/metrics.py`: implements the fixed span-level metrics contract.

## Design rules

1. Character offsets are half-open and document-global at module boundaries.
2. Transformations are pure unless a class owns real model state.
3. Public data is represented by immutable slotted dataclasses, not arbitrary dictionaries.
4. Heavy ML and Presidio imports are lazy or injected.
5. No generic `utils.py`, `helpers.py`, or manager classes.
6. A Python module above 300 lines fails the structure check.
7. Future modules are added with working behaviour and tests, not placeholders.
