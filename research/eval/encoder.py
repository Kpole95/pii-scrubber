"""Load and run the trained encoder for span-level evaluation."""

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from pii_scrub.text import merge_window_predictions, token_predictions_to_spans
from pii_scrub.types import CharacterSpan, DetectedSpan, Offset, WindowSpan

EncoderPredictor = Callable[[str, set[str] | None], list[DetectedSpan]]

_NON_SPACE = re.compile(r"\S+")


def load_encoder_predictor(
    model_path: Path,
    *,
    max_length: int = 512,
    overlap: int = 64,
    device: str | None = None,
) -> EncoderPredictor:
    """Load a trained token classifier and return a detector-compatible predictor.

    Example:
        ``predict = load_encoder_predictor(Path("research/results/encoder"))``
        creates a callable accepted by ``EncoderDetector``.
    """

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("encoder inference requires a fast tokenizer")

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    model.eval()

    id_to_label = {int(index): label for index, label in model.config.id2label.items()}

    def predict(text: str, entities: set[str] | None = None) -> list[DetectedSpan]:
        """Predict document-global PII spans for one text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []

        words, word_offsets = _split_words(text)
        if not words:
            return []

        encoded = tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            stride=overlap,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )

        local_offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        model_inputs = {name: tensor.to(resolved_device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            logits = model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores, label_ids = probabilities.max(dim=-1)

        predictions: list[WindowSpan] = []

        for window_index in range(label_ids.shape[0]):
            word_ids = encoded.word_ids(batch_index=window_index)
            if word_ids is None:
                raise ValueError("fast tokenizer did not return word IDs")

            offsets = _project_offsets(
                [tuple(offset) for offset in local_offsets[window_index].tolist()],
                word_ids,
                word_offsets,
            )
            labels = [id_to_label[int(label_id)] for label_id in label_ids[window_index].tolist()]
            token_scores = [float(score) for score in scores[window_index].tolist()]
            spans = token_predictions_to_spans(offsets, labels, token_scores)

            predictions.extend(
                WindowSpan(
                    window_index,
                    CharacterSpan(span.start, span.end, span.entity_type),
                    span.score,
                )
                for span in spans
                if entities is None or span.entity_type in entities
            )

        return [
            DetectedSpan(item.span.start, item.span.end, item.span.entity_type, item.score)
            for item in merge_window_predictions(predictions)
        ]

    return predict


def _split_words(text: str) -> tuple[tuple[str, ...], tuple[Offset, ...]]:
    """Split source text without losing exact document character offsets."""

    matches = tuple(_NON_SPACE.finditer(text))
    return (
        tuple(match.group() for match in matches),
        tuple((match.start(), match.end()) for match in matches),
    )


def _project_offsets(
    local_offsets: Sequence[Offset],
    word_ids: Sequence[int | None],
    word_offsets: Sequence[Offset],
) -> list[Offset]:
    """Project pre-tokenized subword offsets back onto the source document."""

    if len(local_offsets) != len(word_ids):
        raise ValueError("token offsets and word IDs must contain the same number of items")

    projected: list[Offset] = []

    for index, (local, word_id) in enumerate(zip(local_offsets, word_ids, strict=True)):
        if word_id is None:
            projected.append((0, 0))
            continue
        if word_id < 0 or word_id >= len(word_offsets):
            raise ValueError(f"token at index {index} refers to invalid word index {word_id}")

        word_start, word_end = word_offsets[word_id]
        local_start, local_end = local
        start, end = word_start + local_start, word_start + local_end

        if start < word_start or end > word_end or end < start:
            raise ValueError(f"token offset at index {index} exceeds its source word")

        projected.append((start, end))

    return projected
