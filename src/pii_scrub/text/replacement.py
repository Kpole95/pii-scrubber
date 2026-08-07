"""Replace detected spans with typed placeholders and restore them safely."""

from collections.abc import Sequence
from dataclasses import dataclass

from pii_scrub.errors import InvalidSpanError, RestoreError
from pii_scrub.types import CharacterSpan


@dataclass(frozen=True, slots=True)
class RestoreEntry:
    """Map one generated placeholder to the exact removed source value."""

    placeholder: str
    value: str


@dataclass(frozen=True, slots=True)
class ReplacementResult:
    """Contain redacted text and the caller-owned restoration mapping."""

    text: str
    mapping: tuple[RestoreEntry, ...]


def replace_spans(text: str, spans: Sequence[CharacterSpan]) -> ReplacementResult:
    """Replace sorted non-overlapping spans using deterministic placeholders.

    Example:
        PERSON spans become ``[PERSON_1]``, ``[PERSON_2]``, and so on.
        Replacements are applied right-to-left so original offsets stay valid.
    """

    ordered = sorted(spans, key=lambda item: (item.start, item.end, item.entity_type))
    _validate_spans(text, ordered)
    counts: dict[str, int] = {}
    entries: list[tuple[CharacterSpan, RestoreEntry]] = []
    for span in ordered:
        counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
        placeholder = f"[{span.entity_type}_{counts[span.entity_type]}]"
        entries.append((span, RestoreEntry(placeholder, span.extract(text))))

    redacted = text
    for span, entry in reversed(entries):
        redacted = redacted[: span.start] + entry.placeholder + redacted[span.end :]
    return ReplacementResult(redacted, tuple(entry for _, entry in entries))


def restore_text(text: str, mapping: Sequence[RestoreEntry]) -> str:
    """Restore placeholders exactly once and reject damaged mappings."""

    restored = text
    for entry in mapping:
        if not isinstance(entry, RestoreEntry):
            raise TypeError("mapping must contain RestoreEntry objects")
        occurrences = restored.count(entry.placeholder)
        if occurrences != 1:
            raise RestoreError(
                f"placeholder {entry.placeholder!r} must appear exactly once; found {occurrences}"
            )
        restored = restored.replace(entry.placeholder, entry.value, 1)
    return restored


def _validate_spans(text: str, spans: Sequence[CharacterSpan]) -> None:
    """Validate source bounds and reject overlapping replacements."""

    for index, span in enumerate(spans):
        if not isinstance(span, CharacterSpan):
            raise TypeError("spans must contain CharacterSpan objects")
        if span.end > len(text):
            raise InvalidSpanError(f"span at index {index} exceeds text length")
        if index and span.start < spans[index - 1].end:
            raise InvalidSpanError("replacement spans must not overlap")
