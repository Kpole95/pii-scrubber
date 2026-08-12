"""Tests for OOD JSONL loading."""

import pytest

from pii_scrub.types import CharacterSpan
from research.data.ood_io import load_ood_jsonl


def test_loads_jsonl(tmp_path) -> None:
    """Valid JSONL rows should become normalized examples."""
    path = tmp_path / "ood.jsonl"
    path.write_text(
        '{"id":"001","text":"Hi John","spans":'
        '[{"start":3,"end":7,"label":"PERSON"}]}\n'
        '{"id":"002","text":"No private data","spans":[]}\n',
        encoding="utf-8",
    )

    result = load_ood_jsonl(path)

    assert len(result) == 2
    assert result[0].spans == (CharacterSpan(3, 7, "PERSON"),)
    assert result[1].spans == ()


def test_skips_blank_lines(tmp_path) -> None:
    """Blank JSONL lines should be ignored."""
    path = tmp_path / "ood.jsonl"
    path.write_text(
        '\n{"id":"001","text":"Hello","spans":[]}\n\n',
        encoding="utf-8",
    )

    assert len(load_ood_jsonl(path)) == 1


def test_reports_bad_line_number(tmp_path) -> None:
    """Invalid JSON should report the exact source line."""
    path = tmp_path / "ood.jsonl"
    path.write_text(
        '{"id":"001","text":"Hello","spans":[]}\n{bad json}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"ood\.jsonl:2"):
        load_ood_jsonl(path)


def test_rejects_missing_file(tmp_path) -> None:
    """A missing OOD file should fail clearly."""
    with pytest.raises(FileNotFoundError):
        load_ood_jsonl(tmp_path / "missing.jsonl")
