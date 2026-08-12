# tests / unit / text

Text-processing tests grouped by alignment, span/redaction behavior, and window resolution.

## Folders

- `alignment/` — Tokenizer/BIO alignment tests.
- `spans/` — Span validation and redaction tests.
- `windows/` — Window overlap and merge tests.

## Files

- `__init__.py` — Unit tests for text alignment, spans, windows, and replacement. Main defs/classes: No top-level defs or classes.
- `conftest.py` — Deterministic tokenizer fixtures with no network or ML dependency. Main defs/classes: `FakeEncoding`, `FakeFastTokenizer`, `_build_tokenizer`, `toy_tokenizer`, `unicode_toy_tokenizer`, `limited_toy_tokenizer`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
