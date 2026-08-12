"""Tests for strict Qwen generative output parsing."""

import pytest

from pii_scrub.types import DetectedSpan
from research.eval.qwen import parse_qwen_output


def test_parses_valid_span_output() -> None:
    """Valid generated JSON should become detected spans."""

    assert parse_qwen_output(
        "John emailed me.",
        '{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}',
    ) == [
        DetectedSpan(0, 4, "PERSON"),
    ]


def test_parses_empty_span_output() -> None:
    """No-PII output should produce no predictions."""

    assert (
        parse_qwen_output(
            "Nothing sensitive.",
            '{"spans":[]}',
        )
        == []
    )


def test_allows_json_surrounding_whitespace() -> None:
    """Normal surrounding whitespace should remain valid JSON."""

    assert parse_qwen_output(
        "John",
        '  \n{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}\n',
    ) == [
        DetectedSpan(0, 4, "PERSON"),
    ]


def test_rejects_markdown_fence() -> None:
    """Markdown-wrapped JSON should fail the strict generation contract."""

    with pytest.raises(ValueError, match="valid JSON"):
        parse_qwen_output(
            "John",
            '```json\n{"spans":[]}\n```',
        )


def test_rejects_wrong_top_level_schema() -> None:
    """Only the spans field should exist at the top level."""

    with pytest.raises(ValueError, match="only the 'spans' field"):
        parse_qwen_output(
            "John",
            '{"spans":[],"explanation":"none"}',
        )


def test_rejects_non_integer_offsets() -> None:
    """Generated offsets must be real integers, not strings or booleans."""

    with pytest.raises(ValueError, match="offsets must be integers"):
        parse_qwen_output(
            "John",
            '{"spans":[{"start":false,"end":4,"entity_type":"PERSON"}]}',
        )


def test_rejects_out_of_bounds_span() -> None:
    """Generated spans must stay within the original source text."""

    with pytest.raises(ValueError, match="invalid source offsets"):
        parse_qwen_output(
            "John",
            '{"spans":[{"start":0,"end":10,"entity_type":"PERSON"}]}',
        )


def test_rejects_zero_length_span() -> None:
    """Generated spans must contain at least one source character."""

    with pytest.raises(ValueError, match="invalid source offsets"):
        parse_qwen_output(
            "John",
            '{"spans":[{"start":2,"end":2,"entity_type":"PERSON"}]}',
        )


def test_rejects_unknown_entity_type() -> None:
    """Evaluation may restrict predictions to the known training taxonomy."""

    with pytest.raises(ValueError, match="unsupported entity type"):
        parse_qwen_output(
            "John",
            '{"spans":[{"start":0,"end":4,"entity_type":"ALIEN"}]}',
            entity_types={"PERSON", "EMAIL"},
        )


def test_rejects_unsorted_spans() -> None:
    """Generated spans must follow ascending source order."""

    with pytest.raises(ValueError, match="ascending source order"):
        parse_qwen_output(
            "John called Alice",
            (
                '{"spans":['
                '{"start":12,"end":17,"entity_type":"PERSON"},'
                '{"start":0,"end":4,"entity_type":"PERSON"}'
                "]}"
            ),
        )


def test_rejects_overlapping_spans() -> None:
    """Generated spans must not overlap each other."""

    with pytest.raises(ValueError, match="overlaps"):
        parse_qwen_output(
            "John Smith",
            (
                '{"spans":['
                '{"start":0,"end":10,"entity_type":"PERSON"},'
                '{"start":5,"end":10,"entity_type":"PERSON"}'
                "]}"
            ),
        )


def test_rejects_extra_span_fields() -> None:
    """The model must not invent unsupported span attributes."""

    with pytest.raises(ValueError, match="only start, end, and entity_type"):
        parse_qwen_output(
            "John",
            ('{"spans":[{"start":0,"end":4,"entity_type":"PERSON","score":0.9}]}'),
        )
