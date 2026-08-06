"""Data models for predictions produced from document windows."""

from dataclasses import dataclass

from pii_scrub.data.alignment import CharacterSpan


@dataclass(frozen=True, slots=True)
class WindowSpan:
    """Store one entity prediction produced by one document window.

    The span offsets always refer to the complete original document,
    not to the window's local text.

    Example:
        Window 2 predicts a PERSON at document characters 100–112:

        ``WindowSpan(
            window_index=2,
            span=CharacterSpan(100, 112, "PERSON"),
            score=0.95,
        )``
    """

    window_index: int
    span: CharacterSpan
    score: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_index, bool)
            or not isinstance(self.window_index, int)
        ):
            raise TypeError("window_index must be an integer")

        if self.window_index < 0:
            raise ValueError("window_index must be non-negative")

        if not isinstance(self.span, CharacterSpan):
            raise TypeError("span must be a CharacterSpan")

        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(
                self.score,
                int | float,
            ):
                raise TypeError("score must be a number or None")

            if not 0.0 <= float(self.score) <= 1.0:
                raise ValueError("score must be between 0 and 1")

            object.__setattr__(self, "score", float(self.score))