"""Tests for generative Qwen training examples."""

import json

import pytest

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample
from research.train.qwen_data import (
    SYSTEM_PROMPT,
    build_qwen_messages,
    format_qwen_target,
)


def _example(
    text: str,
    spans: tuple[CharacterSpan, ...],
) -> DatasetExample:
    return DatasetExample(
        example_id="example-1",
        text=text,
        spans=spans,
        source="test",
        language="en",
    )


def test_formats_compact_span_json() -> None:
    """Gold character spans should become deterministic compact JSON."""

    example = _example(
        "John emailed me.",
        (CharacterSpan(0, 4, "PERSON"),),
    )

    assert format_qwen_target(example) == ('{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}')


def test_formats_multiple_spans_in_source_order() -> None:
    """Target serialization should preserve normalized span order."""

    example = _example(
        "John john@example.com",
        (
            CharacterSpan(0, 4, "PERSON"),
            CharacterSpan(5, 21, "EMAIL"),
        ),
    )

    payload = json.loads(format_qwen_target(example))

    assert payload == {
        "spans": [
            {"start": 0, "end": 4, "entity_type": "PERSON"},
            {"start": 5, "end": 21, "entity_type": "EMAIL"},
        ]
    }


def test_formats_no_pii_as_empty_span_list() -> None:
    """Examples without PII should still produce valid schema output."""

    example = _example("Nothing sensitive here.", ())

    assert format_qwen_target(example) == '{"spans":[]}'


def test_preserves_unicode_text() -> None:
    """Chat examples should preserve source Unicode without escaping it."""

    example = _example(
        "Contact José.",
        (CharacterSpan(8, 12, "PERSON"),),
    )

    messages = build_qwen_messages(example)

    assert messages[1]["content"] == "Contact José."
    assert messages[2]["content"] == ('{"spans":[{"start":8,"end":12,"entity_type":"PERSON"}]}')


def test_builds_system_user_assistant_messages() -> None:
    """Training examples should follow the intended chat-role order."""

    example = _example(
        "John",
        (CharacterSpan(0, 4, "PERSON"),),
    )

    messages = build_qwen_messages(example)

    assert tuple(message["role"] for message in messages) == (
        "system",
        "user",
        "assistant",
    )
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "John"
    assert messages[2]["content"] == format_qwen_target(example)


def test_prompt_defines_half_open_offsets() -> None:
    """The model prompt must state the repository's span convention."""

    assert "half-open" in SYSTEM_PROMPT
    assert "inclusive" in SYSTEM_PROMPT
    assert "exclusive" in SYSTEM_PROMPT


def test_rejects_wrong_target_input_type() -> None:
    """Target formatting should require normalized DatasetExample input."""

    with pytest.raises(TypeError, match="DatasetExample"):
        format_qwen_target("John")  # type: ignore[arg-type]


def test_rejects_wrong_message_input_type() -> None:
    """Message building should require normalized DatasetExample input."""

    with pytest.raises(TypeError, match="DatasetExample"):
        build_qwen_messages("John")  # type: ignore[arg-type]
