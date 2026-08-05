from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import pytest

from pii_scrub.data.alignment import (
    AlignedExample,
    CharacterSpan,
    WordOffset,
    align_bio_to_subwords,
    aligned_labels_to_spans,
    bio_tags_to_spans,
    labels_to_ids,
    locate_words,
    parse_bio_label,
)

@pytest.fixture
def toy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a small local WordPiece tokenizer for deterministic tests."""

    vocabulary = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "Call": 4,
        "Mur": 5,
        "##ali": 6,
        "Krishna": 7,
        "today": 8,
        ".": 9,
    }

    backend = Tokenizer(
        WordPiece(
            vocab=vocabulary,
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = BertPreTokenizer()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )




def test_half_open_character_span_extracts_exact_entity() -> None:
    """Verify that [start, end) extracts the exact entity text."""

    text = "Call Murali Krishna today."

    person_start = text.index("Murali")
    person_end = person_start + len("Murali Krishna")

    assert person_start == 5
    assert person_end == 19
    assert text[person_start:person_end] == "Murali Krishna"


def test_character_span_stores_valid_entity_information() -> None:
    """Verify a valid span stores and extracts correct information."""

    text = "Call Murali Krishna today."
    span = CharacterSpan(
        start=5,
        end=19,
        entity_type="PERSON",
    )

    assert span.start == 5
    assert span.end == 19
    assert span.entity_type == "PERSON"
    assert span.length == 14
    assert span.extract(text) == "Murali Krishna"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 5),
        (5, 5),
        (19, 5),
    ],
)
def test_character_span_rejects_invalid_boundaries(
    start: int,
    end: int,
) -> None:
    """Verify negative, empty, and reversed spans are rejected."""

    with pytest.raises(ValueError):
        CharacterSpan(
            start=start,
            end=end,
            entity_type="PERSON",
        )


@pytest.mark.parametrize(
    "entity_type",
    [
        "",
        " ",
        "   ",
    ],
)
def test_character_span_rejects_empty_entity_type(
    entity_type: str,
) -> None:
    """Verify empty or whitespace-only entity types are rejected."""

    with pytest.raises(ValueError):
        CharacterSpan(
            start=5,
            end=19,
            entity_type=entity_type,
        )


def test_character_span_rejects_end_beyond_text() -> None:
    """Verify extraction fails when a span extends beyond the text."""

    text = "Call Murali."
    span = CharacterSpan(
        start=5,
        end=19,
        entity_type="PERSON",
    )

    with pytest.raises(
        ValueError,
        match="span end 19 exceeds text length 12",
    ):
        span.extract(text)


def test_parse_bio_label_parses_outside_label() -> None:
    """Verify that O returns no entity type."""

    prefix, entity_type = parse_bio_label("O")

    assert prefix == "O"
    assert entity_type is None


@pytest.mark.parametrize(
    ("label", "expected_prefix", "expected_entity_type"),
    [
        ("B-PERSON", "B", "PERSON"),
        ("I-EMAIL", "I", "EMAIL"),
        ("B-PHONE_NUMBER", "B", "PHONE_NUMBER"),
        ("I-ID-NUMBER", "I", "ID-NUMBER"),
    ],
)
def test_parse_bio_label_parses_entity_labels(
    label: str,
    expected_prefix: str,
    expected_entity_type: str,
) -> None:
    """Verify valid B and I labels are parsed correctly."""

    prefix, entity_type = parse_bio_label(label)

    assert prefix == expected_prefix
    assert entity_type == expected_entity_type


@pytest.mark.parametrize(
    "label",
    [
        "",
        "PERSON",
        "X-PERSON",
        "O-PERSON",
        "B-",
        "I-",
        " B-PERSON",
        "B-PERSON ",
        "B- PERSON",
    ],
)
def test_parse_bio_label_rejects_malformed_labels(
    label: str,
) -> None:
    """Verify incorrectly formatted BIO labels raise ValueError."""

    with pytest.raises(ValueError):
        parse_bio_label(label)


@pytest.mark.parametrize(
    "label",
    [
        None,
        12,
        True,
    ],
)
def test_parse_bio_label_rejects_non_string_values(
    label: object,
) -> None:
    """Verify BIO labels must be strings."""

    with pytest.raises(TypeError):
        parse_bio_label(label)  # type: ignore[arg-type]


def test_word_offset_extracts_exact_word() -> None:
    """Verify a WordOffset extracts its exact source word."""

    text = "Call Murali Krishna today."
    offset = WordOffset(
        word="Murali",
        start=5,
        end=11,
    )

    assert offset.word == "Murali"
    assert offset.start == 5
    assert offset.end == 11
    assert offset.extract(text) == "Murali"


def test_locate_words_maps_words_to_exact_offsets() -> None:
    """Verify ordinary words map to their exact character positions."""

    text = "Call Murali Krishna today."
    words = ["Call", "Murali", "Krishna", "today", "."]

    offsets = locate_words(text, words)

    assert offsets == [
        WordOffset("Call", 0, 4),
        WordOffset("Murali", 5, 11),
        WordOffset("Krishna", 12, 19),
        WordOffset("today", 20, 25),
        WordOffset(".", 25, 26),
    ]


def test_locate_words_handles_repeated_words_in_order() -> None:
    """Verify repeated words map to separate occurrences."""

    text = "John called John."
    words = ["John", "called", "John", "."]

    offsets = locate_words(text, words)

    assert offsets == [
        WordOffset("John", 0, 4),
        WordOffset("called", 5, 11),
        WordOffset("John", 12, 16),
        WordOffset(".", 16, 17),
    ]


def test_locate_words_preserves_irregular_whitespace() -> None:
    """Verify tabs, newlines, and repeated spaces do not break offsets."""

    text = "Call\tMurali\nKrishna  today."
    words = ["Call", "Murali", "Krishna", "today", "."]

    offsets = locate_words(text, words)

    assert [offset.extract(text) for offset in offsets] == words
    assert [(offset.start, offset.end) for offset in offsets] == [
        (0, 4),
        (5, 11),
        (12, 19),
        (21, 26),
        (26, 27),
    ]


def test_locate_words_handles_unicode_text() -> None:
    """Verify Unicode words receive correct Python character offsets."""

    text = "Contact José Álvarez today."
    words = ["Contact", "José", "Álvarez", "today", "."]

    offsets = locate_words(text, words)

    assert [offset.extract(text) for offset in offsets] == words
    assert offsets[1] == WordOffset("José", 8, 12)
    assert offsets[2] == WordOffset("Álvarez", 13, 20)


def test_locate_words_returns_empty_list_for_no_words() -> None:
    """Verify an empty word list produces an empty result."""

    assert locate_words("No annotations.", []) == []


def test_locate_words_rejects_missing_word() -> None:
    """Verify a word missing from the source text raises ValueError."""

    with pytest.raises(
        ValueError,
        match="word 'Krishna' at index 2 was not found",
    ):
        locate_words(
            text="Call Murali today.",
            words=["Call", "Murali", "Krishna", "today", "."],
        )


def test_locate_words_rejects_words_in_wrong_order() -> None:
    """Verify words must follow their original source-text order."""

    with pytest.raises(ValueError):
        locate_words(
            text="Murali called Krishna.",
            words=["Krishna", "Murali", "called", "."],
        )

def test_bio_tags_to_spans_builds_multiword_person() -> None:
    """A B/I PERSON sequence should become one character span."""

    text = "Call Murali Krishna today."
    words = ["Call", "Murali", "Krishna", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    spans = bio_tags_to_spans(text, words, labels)

    assert spans == [
        CharacterSpan(
            start=5,
            end=19,
            entity_type="PERSON",
        )
    ]
    assert spans[0].extract(text) == "Murali Krishna"


def test_bio_tags_to_spans_builds_multiple_entities() -> None:
    """Separate BIO sequences should become separate character spans."""

    text = "Email Murali at murali@example.com."
    words = [
        "Email",
        "Murali",
        "at",
        "murali@example.com",
        ".",
    ]
    labels = [
        "O",
        "B-PERSON",
        "O",
        "B-EMAIL",
        "O",
    ]

    spans = bio_tags_to_spans(text, words, labels)

    assert spans == [
        CharacterSpan(6, 12, "PERSON"),
        CharacterSpan(16, 34, "EMAIL"),
    ]

    assert [span.extract(text) for span in spans] == [
        "Murali",
        "murali@example.com",
    ]


def test_bio_tags_to_spans_closes_entity_at_end_of_text() -> None:
    """An entity ending on the last word must still be stored."""

    text = "Contact Murali Krishna"
    words = ["Contact", "Murali", "Krishna"]
    labels = ["O", "B-PERSON", "I-PERSON"]

    spans = bio_tags_to_spans(text, words, labels)

    assert spans == [
        CharacterSpan(8, 22, "PERSON"),
    ]


def test_bio_tags_to_spans_handles_adjacent_entities() -> None:
    """A new B label should close the previous entity immediately."""

    text = "Murali London"
    words = ["Murali", "London"]
    labels = ["B-PERSON", "B-LOCATION"]

    spans = bio_tags_to_spans(text, words, labels)

    assert spans == [
        CharacterSpan(0, 6, "PERSON"),
        CharacterSpan(7, 13, "LOCATION"),
    ]


def test_bio_tags_to_spans_returns_empty_list_without_entities() -> None:
    """An all-O sequence should produce no entity spans."""

    text = "Nothing private here."
    words = ["Nothing", "private", "here", "."]
    labels = ["O", "O", "O", "O"]

    assert bio_tags_to_spans(text, words, labels) == []


def test_bio_tags_to_spans_rejects_different_lengths() -> None:
    """Every word must have exactly one BIO label."""

    with pytest.raises(
        ValueError,
        match="words and labels must contain the same number of items",
    ):
        bio_tags_to_spans(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "B-PERSON"],
        )


def test_bio_tags_to_spans_rejects_orphan_i_label() -> None:
    """An I label cannot appear without a preceding matching B label."""

    with pytest.raises(
        ValueError,
        match="I-label at index 1 has no active entity",
    ):
        bio_tags_to_spans(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "I-PERSON", "O"],
        )


def test_bio_tags_to_spans_rejects_type_change_inside_entity() -> None:
    """An I label must match the type of the active entity."""

    with pytest.raises(
        ValueError,
        match="active entity has type 'PERSON'",
    ):
        bio_tags_to_spans(
            text="Murali Krishna",
            words=["Murali", "Krishna"],
            labels=["B-PERSON", "I-LOCATION"],
        )


def test_aligned_example_stores_token_level_information() -> None:
    """An aligned example should store matching token-level fields."""

    example = AlignedExample(
        input_ids=(101, 2000, 102),
        attention_mask=(1, 1, 1),
        offset_mapping=((0, 0), (0, 4), (0, 0)),
        word_ids=(None, 0, None),
        token_labels=(None, "O", None),
    )

    assert example.input_ids == (101, 2000, 102)
    assert example.attention_mask == (1, 1, 1)
    assert example.offset_mapping == ((0, 0), (0, 4), (0, 0))
    assert example.word_ids == (None, 0, None)
    assert example.token_labels == (None, "O", None)
    assert example.token_count == 3


def test_aligned_example_rejects_different_field_lengths() -> None:
    """Every token must have a mask, offset, word ID, and label."""

    with pytest.raises(
        ValueError,
        match="all AlignedExample fields must have equal lengths",
    ):
        AlignedExample(
            input_ids=(101, 2000, 102),
            attention_mask=(1, 1),
            offset_mapping=((0, 0), (0, 4), (0, 0)),
            word_ids=(None, 0, None),
            token_labels=(None, "O", None),
        )


def test_aligned_example_rejects_empty_token_sequence() -> None:
    """An aligned example must contain at least one token."""

    with pytest.raises(
        ValueError,
        match="AlignedExample must contain at least one token",
    ):
        AlignedExample(
            input_ids=(),
            attention_mask=(),
            offset_mapping=(),
            word_ids=(),
            token_labels=(),
        )


def test_aligned_example_is_immutable() -> None:
    """Tokenized training data should not change after validation."""

    example = AlignedExample(
        input_ids=(101, 2000, 102),
        attention_mask=(1, 1, 1),
        offset_mapping=((0, 0), (0, 4), (0, 0)),
        word_ids=(None, 0, None),
        token_labels=(None, "O", None),
    )

    with pytest.raises(AttributeError):
        example.input_ids = (999,)  # type: ignore[misc]

def test_align_bio_to_subwords_propagates_labels(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """A B label should become I on later subwords of the same word."""

    text = "Call Murali Krishna today."
    words = ["Call", "Murali", "Krishna", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    example = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
    )

    assert example.token_labels == (
        None,
        "O",
        "B-PERSON",
        "I-PERSON",
        "I-PERSON",
        "O",
        "O",
        None,
    )


def test_align_bio_to_subwords_returns_original_text_offsets(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Subword offsets should refer to the original unsplit text."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.offset_mapping == (
        (0, 0),    # [CLS]
        (0, 4),    # Call
        (5, 8),    # Mur
        (8, 11),   # ##ali
        (12, 19),  # Krishna
        (20, 25),  # today
        (25, 26),  # .
        (0, 0),    # [SEP]
    )


def test_align_bio_to_subwords_preserves_word_ids(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Every subword should retain its source-word index."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.word_ids == (
        None,
        0,
        1,
        1,
        2,
        3,
        4,
        None,
    )


def test_align_bio_to_subwords_special_tokens_have_no_labels(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Special tokens should not receive training BIO labels."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.word_ids[0] is None
    assert example.token_labels[0] is None
    assert example.offset_mapping[0] == (0, 0)

    assert example.word_ids[-1] is None
    assert example.token_labels[-1] is None
    assert example.offset_mapping[-1] == (0, 0)


def test_align_bio_to_subwords_rejects_different_lengths(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Every source word must have exactly one BIO label."""

    with pytest.raises(
        ValueError,
        match="words and labels must contain the same number of items",
    ):
        align_bio_to_subwords(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "B-PERSON"],
            tokenizer=toy_tokenizer,
        )

def test_align_bio_to_subwords_first_subword_strategy(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Only the first subword should receive a label."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
        strategy="first_subword",
    )

    assert example.token_labels == (
        None,          # [CLS]
        "O",           # Call
        "B-PERSON",    # Mur
        None,          # ##ali
        "I-PERSON",    # Krishna
        "O",           # today
        "O",           # .
        None,          # [SEP]
    )


def test_align_bio_to_subwords_all_subwords_is_default(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """The default strategy should label every subword."""

    example = align_bio_to_subwords(
        text="Call Murali Krishna today.",
        words=["Call", "Murali", "Krishna", "today", "."],
        labels=["O", "B-PERSON", "I-PERSON", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert example.token_labels[2:4] == (
        "B-PERSON",
        "I-PERSON",
    )

def test_align_bio_to_subwords_first_strategy_ignores_later_o_subwords(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Later subwords of O-labeled words should also be ignored."""

    example = align_bio_to_subwords(
        text="Murali",
        words=["Murali"],
        labels=["O"],
        tokenizer=toy_tokenizer,
        strategy="first_subword",
    )

    assert example.token_labels == (
        None,
        "O",
        None,
        None,
    )

def test_align_bio_to_subwords_rejects_unknown_strategy(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Unsupported subword strategies must be rejected."""

    with pytest.raises(
        ValueError,
        match="strategy must be 'all_subwords' or 'first_subword'",
    ):
        align_bio_to_subwords(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "B-PERSON", "O"],
            tokenizer=toy_tokenizer,
            strategy="unknown",  # type: ignore[arg-type]
        )

def test_aligned_example_allows_ignored_source_token_label() -> None:
    """A source subword may use None when excluded from training loss."""

    example = AlignedExample(
        input_ids=(1, 5, 6, 2),
        attention_mask=(1, 1, 1, 1),
        offset_mapping=((0, 0), (5, 8), (8, 11), (0, 0)),
        word_ids=(None, 0, 0, None),
        token_labels=(None, "B-PERSON", None, None),
    )

    assert example.token_labels == (
        None,
        "B-PERSON",
        None,
        None,
    )

def test_labels_to_ids_converts_bio_labels() -> None:
    """BIO strings should be converted into their numeric class IDs."""

    result = labels_to_ids(
        token_labels=(
            None,
            "O",
            "B-PERSON",
            None,
            "I-PERSON",
            None,
        ),
        label_to_id={
            "O": 0,
            "B-PERSON": 1,
            "I-PERSON": 2,
        },
    )

    assert result == (
        -100,
        0,
        1,
        -100,
        2,
        -100,
    )

def test_labels_to_ids_supports_custom_ignore_index() -> None:
    """The caller should be able to choose another ignore value."""

    result = labels_to_ids(
        token_labels=("O", None),
        label_to_id={"O": 0},
        ignore_index=-1,
    )

    assert result == (0, -1)

def test_labels_to_ids_rejects_missing_label_mapping() -> None:
    """Every non-ignored token label must exist in the label mapping."""

    with pytest.raises(
        ValueError,
        match="label 'B-EMAIL' at token index 1 is missing",
    ):
        labels_to_ids(
            token_labels=("O", "B-EMAIL"),
            label_to_id={"O": 0},
        )
    
def test_labels_to_ids_rejects_duplicate_ids() -> None:
    """Different BIO labels must not share the same model class ID."""

    with pytest.raises(
        ValueError,
        match="label IDs must be unique",
    ):
        labels_to_ids(
            token_labels=("O", "B-PERSON"),
            label_to_id={
                "O": 0,
                "B-PERSON": 0,
            },
        )

def test_labels_to_ids_reserves_ignore_index() -> None:
    """No real BIO class may use the ignored-token ID."""

    with pytest.raises(
        ValueError,
        match="must not equal ignore_index",
    ):
        labels_to_ids(
            token_labels=("O",),
            label_to_id={"O": -100},
        )

def test_all_subword_alignment_round_trips_to_original_spans(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """All-subword alignment should reconstruct the original gold spans."""

    text = "Call Murali Krishna today."
    words = ["Call", "Murali", "Krishna", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    gold_spans = bio_tags_to_spans(text, words, labels)

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
        strategy="all_subwords",
    )

    reconstructed_spans = aligned_labels_to_spans(aligned)

    assert reconstructed_spans == gold_spans
    assert reconstructed_spans == [
        CharacterSpan(5, 19, "PERSON"),
    ]

def test_first_subword_alignment_round_trips_multiword_entity(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """First-subword alignment should preserve a multiword entity span."""

    text = "Call Murali Krishna today."
    words = ["Call", "Murali", "Krishna", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    gold_spans = bio_tags_to_spans(text, words, labels)

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
        strategy="first_subword",
    )

    reconstructed_spans = aligned_labels_to_spans(aligned)

    assert reconstructed_spans == gold_spans

def test_aligned_labels_to_spans_builds_multiple_entities(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Aligned labels should reconstruct separate entity spans."""

    text = "Murali Krishna"
    words = ["Murali", "Krishna"]
    labels = ["B-PERSON", "B-LOCATION"]

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
    )

    spans = aligned_labels_to_spans(aligned)

    assert spans == [
        CharacterSpan(0, 6, "PERSON"),
        CharacterSpan(7, 14, "LOCATION"),
    ]

def test_aligned_labels_to_spans_returns_empty_list_for_all_o(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """An all-O aligned sequence should reconstruct no spans."""

    aligned = align_bio_to_subwords(
        text="Call today.",
        words=["Call", "today", "."],
        labels=["O", "O", "O"],
        tokenizer=toy_tokenizer,
    )

    assert aligned_labels_to_spans(aligned) == []

def test_aligned_labels_to_spans_rejects_orphan_i_label() -> None:
    """An aligned I label cannot exist without an active entity."""

    example = AlignedExample(
        input_ids=(1, 5, 2),
        attention_mask=(1, 1, 1),
        offset_mapping=((0, 0), (0, 3), (0, 0)),
        word_ids=(None, 0, None),
        token_labels=(None, "I-PERSON", None),
    )

    with pytest.raises(
        ValueError,
        match="I-label at token index 1 has no active entity",
    ):
        aligned_labels_to_spans(example)

def test_aligned_labels_to_spans_rejects_type_change() -> None:
    """An I label must match the active entity type."""

    example = AlignedExample(
        input_ids=(1, 5, 7, 2),
        attention_mask=(1, 1, 1, 1),
        offset_mapping=((0, 0), (0, 3), (4, 11), (0, 0)),
        word_ids=(None, 0, 1, None),
        token_labels=(
            None,
            "B-PERSON",
            "I-LOCATION",
            None,
        ),
    )

    with pytest.raises(
        ValueError,
        match="active entity has type 'PERSON'",
    ):
        aligned_labels_to_spans(example)

def test_first_subword_alignment_round_trips_split_single_word_entity(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Ignored continuation subwords must preserve the complete word span."""

    text = "Call Murali today."
    words = ["Call", "Murali", "today", "."]
    labels = ["O", "B-PERSON", "O", "O"]

    gold_spans = bio_tags_to_spans(
        text=text,
        words=words,
        labels=labels,
    )

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
        strategy="first_subword",
    )

    reconstructed = aligned_labels_to_spans(aligned)

    assert aligned.token_labels == (
        None,
        "O",
        "B-PERSON",
        None,
        "O",
        "O",
        None,
    )

    assert reconstructed == gold_spans
    assert reconstructed == [
        CharacterSpan(5, 11, "PERSON"),
    ]
    assert reconstructed[0].extract(text) == "Murali"

def test_ignored_special_tokens_do_not_extend_entity_span() -> None:
    """A special token with word_id None must not change entity boundaries."""

    example = AlignedExample(
        input_ids=(1, 5, 2),
        attention_mask=(1, 1, 1),
        offset_mapping=((0, 0), (5, 11), (0, 0)),
        word_ids=(None, 0, None),
        token_labels=(None, "B-PERSON", None),
    )

    spans = aligned_labels_to_spans(example)

    assert spans == [
        CharacterSpan(5, 11, "PERSON"),
    ]

def test_round_trip_preserves_multiple_entities(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Multiple entities should survive alignment and reconstruction."""

    text = "Murali met Krishna."
    words = ["Murali", "met", "Krishna", "."]
    labels = [
        "B-PERSON",
        "O",
        "B-PERSON",
        "O",
    ]

    gold_spans = bio_tags_to_spans(text, words, labels)

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
        strategy="all_subwords",
    )

    reconstructed = aligned_labels_to_spans(aligned)

    assert reconstructed == gold_spans
    assert [span.extract(text) for span in reconstructed] == [
        "Murali",
        "Krishna",
    ]

def test_round_trip_preserves_offsets_with_irregular_whitespace(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Tabs, newlines, and repeated spaces must not corrupt offsets."""

    text = "Call\tMurali\nKrishna  today."
    words = ["Call", "Murali", "Krishna", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    gold_spans = bio_tags_to_spans(text, words, labels)

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
        strategy="all_subwords",
    )

    reconstructed = aligned_labels_to_spans(aligned)

    assert reconstructed == gold_spans
    assert reconstructed == [
        CharacterSpan(5, 19, "PERSON"),
    ]
    assert reconstructed[0].extract(text) == "Murali\nKrishna"

def test_first_subword_round_trip_preserves_irregular_whitespace(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """First-subword alignment must preserve original entity boundaries."""

    text = "Call\tMurali\nKrishna  today."
    words = ["Call", "Murali", "Krishna", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    gold_spans = bio_tags_to_spans(text, words, labels)

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=toy_tokenizer,
        strategy="first_subword",
    )

    reconstructed = aligned_labels_to_spans(aligned)

    assert reconstructed == gold_spans
    assert reconstructed[0].extract(text) == "Murali\nKrishna"

@pytest.fixture
def unicode_toy_tokenizer() -> PreTrainedTokenizerFast:
    """Create a local tokenizer containing Unicode name tokens."""

    vocabulary = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "Contact": 4,
        "José": 5,
        "Álvarez": 6,
        "today": 7,
        ".": 8,
    }

    backend = Tokenizer(
        WordPiece(
            vocab=vocabulary,
            unk_token="[UNK]",
        )
    )

    backend.pre_tokenizer = BertPreTokenizer()

    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )

def test_round_trip_preserves_unicode_entity(
    unicode_toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Unicode names must retain exact Python character offsets."""

    text = "Contact José Álvarez today."
    words = ["Contact", "José", "Álvarez", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    gold_spans = bio_tags_to_spans(text, words, labels)

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=unicode_toy_tokenizer,
        strategy="all_subwords",
    )

    reconstructed = aligned_labels_to_spans(aligned)

    assert reconstructed == gold_spans
    assert reconstructed == [
        CharacterSpan(8, 20, "PERSON"),
    ]
    assert reconstructed[0].extract(text) == "José Álvarez"

def test_round_trip_preserves_unicode_entity(
    unicode_toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Unicode names must retain exact Python character offsets."""

    text = "Contact José Álvarez today."
    words = ["Contact", "José", "Álvarez", "today", "."]
    labels = ["O", "B-PERSON", "I-PERSON", "O", "O"]

    gold_spans = bio_tags_to_spans(text, words, labels)

    aligned = align_bio_to_subwords(
        text=text,
        words=words,
        labels=labels,
        tokenizer=unicode_toy_tokenizer,
        strategy="all_subwords",
    )

    reconstructed = aligned_labels_to_spans(aligned)

    assert reconstructed == gold_spans
    assert reconstructed == [
        CharacterSpan(8, 20, "PERSON"),
    ]
    assert reconstructed[0].extract(text) == "José Álvarez"

def test_align_bio_to_subwords_accepts_input_within_max_length(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """Tokenization should succeed when it fits the requested limit."""

    example = align_bio_to_subwords(
        text="Call Murali.",
        words=["Call", "Murali", "."],
        labels=["O", "B-PERSON", "O"],
        tokenizer=toy_tokenizer,
        max_length=6,
    )

    assert example.token_count == 6

def test_align_bio_to_subwords_rejects_overlong_input(
    toy_tokenizer: PreTrainedTokenizerFast,
) -> None:
    """An overlong example must fail instead of being truncated."""

    with pytest.raises(
        ValueError,
        match=(
            r"tokenized example contains 6 tokens, "
            r"which exceeds max_length=5"
        ),
    ):
        align_bio_to_subwords(
            text="Call Murali.",
            words=["Call", "Murali", "."],
            labels=["O", "B-PERSON", "O"],
            tokenizer=toy_tokenizer,
            max_length=5,
        )

@pytest.mark.parametrize(
    "max_length",
    [
        0,
        -1,
    ],
)
def test_align_bio_to_subwords_rejects_non_positive_max_length(
    toy_tokenizer: PreTrainedTokenizerFast,
    max_length: int,
) -> None:
    """The maximum token length must be positive."""

    with pytest.raises(
        ValueError,
        match="max_length must be greater than zero",
    ):
        align_bio_to_subwords(
            text="Call.",
            words=["Call", "."],
            labels=["O", "O"],
            tokenizer=toy_tokenizer,
            max_length=max_length,
        )

@pytest.mark.parametrize(
    "max_length",
    [
        5.5,
        "5",
        True,
    ],
)
def test_align_bio_to_subwords_rejects_invalid_max_length_type(
    toy_tokenizer: PreTrainedTokenizerFast,
    max_length: object,
) -> None:
    """The maximum length must be an integer or None."""

    with pytest.raises(
        TypeError,
        match="max_length must be an integer or None",
    ):
        align_bio_to_subwords(
            text="Call.",
            words=["Call", "."],
            labels=["O", "O"],
            tokenizer=toy_tokenizer,
            max_length=max_length,  # type: ignore[arg-type]
        )