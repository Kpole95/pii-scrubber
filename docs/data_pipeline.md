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
