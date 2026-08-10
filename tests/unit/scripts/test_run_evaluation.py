"""Tests for the detector evaluation CLI."""

from pathlib import Path

import pytest

from scripts.run_evaluation import _build_detector, _parser


def test_parser_accepts_required_arguments(tmp_path: Path) -> None:
    """The CLI should parse one complete evaluation run."""

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "report.json"

    args = _parser().parse_args(
        [
            "--detector",
            "regex",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert args.detector == "regex"
    assert args.input == input_path
    assert args.output == output_path


def test_parser_accepts_artifact_split(tmp_path: Path) -> None:
    """The CLI should accept an artifact dataset split."""

    args = _parser().parse_args(
        [
            "--detector",
            "regex",
            "--dataset",
            "artifact",
            "--input",
            str(tmp_path / "artifact"),
            "--split",
            "test",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.dataset == "artifact"
    assert args.split == "test"


def test_parser_accepts_conll_source(tmp_path: Path) -> None:
    """The CLI should accept a CoNLL parquet source."""

    args = _parser().parse_args(
        [
            "--detector",
            "regex",
            "--dataset",
            "conll",
            "--data-file",
            "example.parquet",
            "--split",
            "train",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.dataset == "conll"
    assert args.data_file == "example.parquet"


def test_parser_accepts_encoder_model(tmp_path: Path) -> None:
    """The CLI should accept an encoder model path and device."""

    model_path = tmp_path / "encoder"

    args = _parser().parse_args(
        [
            "--detector",
            "encoder",
            "--model-path",
            str(model_path),
            "--device",
            "cpu",
            "--input",
            str(tmp_path / "input.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.detector == "encoder"
    assert args.model_path == model_path
    assert args.device == "cpu"


def test_encoder_requires_model_path() -> None:
    """Encoder evaluation should fail clearly without model weights."""

    with pytest.raises(ValueError, match="--model-path"):
        _build_detector("encoder", None, None)


def test_parser_accepts_person_merge_ablation(tmp_path: Path) -> None:
    """Encoder evaluation should accept the split-span merge ablation."""

    args = _parser().parse_args(
        [
            "--detector",
            "encoder",
            "--model-path",
            str(tmp_path / "encoder"),
            "--merge-person-fragments",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.merge_person_fragments is True


def test_non_encoder_rejects_person_merge_ablation() -> None:
    """The merge ablation should not apply to non-encoder detectors."""

    with pytest.raises(ValueError, match="supported only for encoder"):
        _build_detector(
            "regex",
            None,
            None,
            merge_person_fragments=True,
        )
