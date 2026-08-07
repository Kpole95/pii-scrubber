"""Convert CoNLL-2003 rows into PERSON-only OOD dataset examples."""

from collections.abc import Mapping, Sequence
from hashlib import sha1
from typing import Any, Final

from pii_scrub.types import CharacterSpan
from research.data.models import DatasetExample

SOURCE: Final = "conll2003"

TAG_BY_ID: Final = {
    0: "O",
    1: "B-ORG",
    2: "B-MISC",
    3: "B-PER",
    4: "I-PER",
    5: "B-LOC",
    6: "I-ORG",
    7: "I-MISC",
    8: "I-LOC",
}

ATTACH_LEFT: Final = {
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "%",
    ")",
    "]",
    "}",
    "'s",
    "'re",
    "'ve",
    "'ll",
    "'d",
    "'m",
    "n't",
}
NO_SPACE_AFTER: Final = {"(", "[", "{", "$", "#"}


def load_conll2003_record(record: Mapping[str, Any]) -> DatasetExample:
    """Convert one CoNLL row into a PERSON-only normalized example.

    Example:
        ``Peter Blackburn`` tagged B-PER/I-PER becomes one PERSON span.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    tokens = tuple(_tokens(record))
    tags = tuple(_tags(record))

    if not tokens:
        raise ValueError("tokens must not be empty")
    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must have equal lengths")

    text, offsets = _detokenize(tokens)

    return DatasetExample(
        example_id=f"conll2003-{_record_id(record, tokens)}",
        text=text,
        spans=_person_spans(tags, offsets),
        source=SOURCE,
        language="en",
    )


def _tokens(record: Mapping[str, Any]) -> Sequence[str]:
    """Validate and return CoNLL tokens.

    Example:
        ``["Peter", "Blackburn", "."]`` is returned unchanged.
    """
    values = record.get("tokens")

    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError("tokens must be a sequence")

    if any(not isinstance(token, str) or not token for token in values):
        raise ValueError("tokens must contain non-empty strings")

    return values


def _tags(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Convert numeric or textual CoNLL tags into tag names.

    Example:
        ``[3, 4, 0]`` becomes ``("B-PER", "I-PER", "O")``.
    """
    values = record.get("tags")

    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError("tags must be a sequence")

    tags: list[str] = []

    for value in values:
        if isinstance(value, bool):
            raise TypeError("tags must contain integers or strings")

        if isinstance(value, int):
            if value not in TAG_BY_ID:
                raise ValueError(f"unsupported CoNLL tag ID: {value}")
            tags.append(TAG_BY_ID[value])
            continue

        tag = value.strip().upper() if isinstance(value, str) else ""
        if tag not in TAG_BY_ID.values():
            raise ValueError(f"unsupported CoNLL tag: {value!r}")
        tags.append(tag)

    return tuple(tags)


def _detokenize(tokens: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Rebuild readable text while recording each token's offsets.

    Example:
        ``["John", "'s", "team", "."]`` becomes ``"John's team."``.
    """
    text = ""
    offsets: list[tuple[int, int]] = []
    previous: str | None = None

    for token in tokens:
        separator = "" if not text or token in ATTACH_LEFT or previous in NO_SPACE_AFTER else " "
        start = len(text) + len(separator)
        text += separator + token
        offsets.append((start, len(text)))
        previous = token

    return text, tuple(offsets)


def _person_spans(
    tags: Sequence[str],
    offsets: Sequence[tuple[int, int]],
) -> tuple[CharacterSpan, ...]:
    """Collapse consecutive B-PER/I-PER tags into PERSON spans.

    Example:
        ``B-PER, I-PER, O`` over ``John Smith left`` gives one PERSON span.
    """
    spans: list[CharacterSpan] = []
    start: int | None = None
    end: int | None = None

    for tag, (token_start, token_end) in zip(tags, offsets, strict=True):
        if tag == "B-PER":
            if start is not None and end is not None:
                spans.append(CharacterSpan(start, end, "PERSON"))
            start, end = token_start, token_end
        elif tag == "I-PER":
            start = token_start if start is None else start
            end = token_end
        elif start is not None and end is not None:
            spans.append(CharacterSpan(start, end, "PERSON"))
            start = end = None

    if start is not None and end is not None:
        spans.append(CharacterSpan(start, end, "PERSON"))

    return tuple(spans)


def _record_id(record: Mapping[str, Any], tokens: Sequence[str]) -> str:
    """Return the source ID or a stable hash when no ID exists.

    Example:
        A row without ``id`` receives the same hash for the same tokens.
    """
    value = record.get("id")

    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise TypeError("id must be an integer or string")
        if result := str(value).strip():
            return result

    return sha1("\x1f".join(tokens).encode(), usedforsecurity=False).hexdigest()[:12]
