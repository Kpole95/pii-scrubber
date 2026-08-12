# tests / integration

End-to-end tests for the public Python API and command-line interface.

## Files

- `__init__.py` — Integration tests for the public API and command-line interface. Main defs/classes: No top-level defs or classes.
- `test_api.py` — Integration tests for the public scrubber orchestration API. Main defs/classes: `test_default_scrubber_redacts_and_restores_structured_pii`, `test_scrubber_applies_per_entity_thresholds`.
- `test_cli.py` — Integration tests for the command-line interface. Main defs/classes: `test_cli_redacts_text_and_writes_mapping`.

## Notes

Keep this guide short. Update it when files move, are added, or change responsibility.
