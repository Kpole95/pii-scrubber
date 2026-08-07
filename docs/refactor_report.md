# Refactor report

## Removed

The uploaded ZIP contained an entire Windows `.venv`, `.pytest_cache`, `.ruff_cache`, `__pycache__`, compiled bytecode, egg metadata, and duplicate root files. None belongs in source control or a handoff archive.

The old nested runtime path `src/pii_scrub/data/{alignment,datasets,windowing}` was removed. Dataset research code no longer ships inside the runtime package.

## Moved

| Old responsibility | Canonical location |
|---|---|
| Alignment models | `src/pii_scrub/types.py` |
| BIO/subword alignment | `src/pii_scrub/text/alignment.py` and `spans.py` |
| Window prediction merging | `src/pii_scrub/text/windows.py` |
| Dataset models/loaders/splits/artifacts/statistics | `research/data/` |
| Span metrics | `research/eval/metrics.py` |

## Added

- reversible typed replacement;
- regex, Presidio, encoder, and generative detector adapters;
- public `Scrubber` API and CLI;
- explicit runtime/research dependency boundary;
- offline tokenizer tests;
- structure enforcement script;
- modular configuration and release documentation.

## Verification

The final suite contains 282 passing tests. The runtime package imports without Transformers, Tokenizers, Torch, Presidio, or network access.
