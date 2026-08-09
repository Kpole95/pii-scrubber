# Data pipeline

## Source normalization

`research/data/openpii.py` validates `source_text`, `uid`, `language`, and each privacy-mask annotation. The annotated `value` must equal the exact source slice.

`research/data/gretel.py` accepts decoded `pii_spans` or JSON text, validates offsets, and normalizes either `label` or `type`.

Both adapters return `DatasetExample` values with sorted, non-overlapping `CharacterSpan` annotations.

## BIO and subword alignment

`align_bio_to_subwords()` accepts a structural fast-tokenizer interface. It preserves global offsets, rejects truncation, verifies complete character coverage, and supports:

- `all_subwords`: continuation pieces receive `I-<TYPE>`;
- `first_subword`: continuation pieces receive `None` and are ignored by loss.

The gate is an exact round trip:

```text
text -> word BIO -> subwords -> aligned BIO -> character spans
```

## Persistence

Deterministic splits are saved as `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `manifest.json`. Loading validates artifact version and every split count.

## Canonical benchmark datasets

Stage 4 adds two canonical dataset adapters:

- `research/data/ai4privacy.py` converts `ai4privacy/pii-masking-200k`
  records into normalized `DatasetExample` objects using original
  character-offset annotations.
- `research/data/conll.py` converts CoNLL-2003 token/tag rows into
  deterministic text with PERSON-only character spans.

Ai4Privacy's pre-generated token/BIO labels are deliberately ignored.
The project uses the original character spans and passes them through
`pii_scrub.text.alignment`, so tokenization and subword alignment remain
under our control.

CoNLL-2003 is used only as a PERSON out-of-distribution benchmark.
ORG, LOC, and MISC annotations are not remapped into unrelated PII
categories.

The data flow is:

```text
raw dataset row
    -> dataset adapter
    -> DatasetExample
    -> CharacterSpan
    -> Stage 3 BIO/subword alignment
    -> training or evaluation