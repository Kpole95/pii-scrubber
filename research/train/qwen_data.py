"""Prepare deterministic chat examples for generative PII training."""

import json
from typing import TypedDict

from research.data.models import DatasetExample


class QwenMessage(TypedDict):
    """One chat-template message."""

    role: str
    content: str


SYSTEM_PROMPT = """You detect personally identifiable information in text.
Return JSON only using this schema:
{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}

Rules:
- Offsets are zero-based character offsets into the exact input text.
- Spans use half-open intervals: start is inclusive and end is exclusive.
- Return spans in ascending source order.
- Use only entity types learned from the training data.
- Do not include explanations, markdown, or copied PII values.
- If no PII exists, return {"spans":[]}."""


def format_qwen_target(example: DatasetExample) -> str:
    """Serialize gold spans into deterministic compact JSON.

    Example:
        A PERSON span ``[0, 4)`` becomes
        ``{"spans":[{"start":0,"end":4,"entity_type":"PERSON"}]}``.
    """

    if not isinstance(example, DatasetExample):
        raise TypeError("example must be a DatasetExample")

    payload = {
        "spans": [
            {
                "start": span.start,
                "end": span.end,
                "entity_type": span.entity_type,
            }
            for span in example.spans
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_qwen_messages(example: DatasetExample) -> tuple[QwenMessage, ...]:
    """Convert one normalized example into system/user/assistant messages."""

    if not isinstance(example, DatasetExample):
        raise TypeError("example must be a DatasetExample")

    return (
        QwenMessage(role="system", content=SYSTEM_PROMPT),
        QwenMessage(role="user", content=example.text),
        QwenMessage(role="assistant", content=format_qwen_target(example)),
    )
