"""Integration tests for the command-line interface."""

import json
from pathlib import Path

from pii_scrub.cli import main


def test_cli_redacts_text_and_writes_mapping(tmp_path: Path, capsys) -> None:
    """Check that the CLI redacts text and writes a restore mapping."""
    mapping = tmp_path / "mapping.json"
    assert main(["Email ana@example.com", "--mapping-out", str(mapping)]) == 0
    assert capsys.readouterr().out.strip() == "Email [EMAIL_1]"
    assert json.loads(mapping.read_text(encoding="utf-8")) == [
        {"placeholder": "[EMAIL_1]", "value": "ana@example.com"}
    ]
