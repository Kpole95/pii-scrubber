"""Shared type aliases for alignment code."""

from typing import Literal, TypeAlias

BioPrefix: TypeAlias = Literal["B", "I", "O"]

SubwordLabelStrategy: TypeAlias = Literal[
    "all_subwords",
    "first_subword",
]

Offset: TypeAlias = tuple[int, int]
TokenLabel: TypeAlias = str | None