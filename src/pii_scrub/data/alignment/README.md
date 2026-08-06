# BIO-to-subword alignment

This package converts strict word-level BIO annotations into token-level labels for a Hugging Face **fast tokenizer**, while retaining half-open character offsets into the original, unnormalised text.

## Pipeline

1. `locate_words` maps annotated words to original-text offsets.
2. `bio_tags_to_spans` builds gold character spans.
3. `align_bio_to_subwords` tokenizes pre-split words and converts local subword offsets to global offsets.
4. `aligned_labels_to_spans` reconstructs spans for round-trip verification.
5. `labels_to_ids` maps labels to training IDs and uses `IGNORE_INDEX` for ignored tokens.

## Strategies

- `all_subwords`: every subword is labelled; continuation pieces use `I-<TYPE>`.
- `first_subword`: only the first subword is labelled. Continuation pieces use `None` and are still included during reconstruction.

## Example

```python
from pii_scrub.data.alignment import align_bio_to_subwords

aligned = align_bio_to_subwords(
    text="Call Murali Krishna today.",
    words=["Call", "Murali", "Krishna", "today", "."],
    labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
    tokenizer=tokenizer,
)
```

## Guarantees and deliberate failures

The package rejects malformed BIO sequences, unmatched source words, unknown-token conversion of annotated entities, incomplete/gapped tokenizer offsets, silent truncation, and over-length examples. Window long inputs before alignment.
