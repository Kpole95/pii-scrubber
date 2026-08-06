"""Map annotated words to exact source-text offsets."""

from collections.abc import Sequence

from .models import WordOffset


def locate_words(
    text: str,
    words: Sequence[str],
) -> list[WordOffset]:
    """Locate annotated words sequentially in the original text.

    Searching continues after the previous word, so repeated words are
    handled correctly.

    Example:
        In ``"John called John."``, the two ``"John"`` words receive
        different offsets.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(words, Sequence) or isinstance(words, str):
        raise TypeError("words must be a sequence of strings")

    offsets: list[WordOffset] = []
    search_start = 0

    for word_index, word in enumerate(words):
        if not isinstance(word, str):
            raise TypeError(
                f"word at index {word_index} must be a string"
            )

        if not word:
            raise ValueError(
                f"word at index {word_index} must not be empty"
            )

        start = text.find(word, search_start)

        if start == -1:
            raise ValueError(
                f"word {word!r} at index {word_index} was not found "
                f"after character {search_start}"
            )

        end = start + len(word)

        offset = WordOffset(
            word=word,
            start=start,
            end=end,
        )

        offset.extract(text)
        offsets.append(offset)
        search_start = end

    return offsets