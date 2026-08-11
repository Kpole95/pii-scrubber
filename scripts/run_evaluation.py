"""Run one detector evaluation from CLI arguments."""

from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from pathlib import Path

from datasets import load_dataset

from pii_scrub.detectors.base import Detector
from pii_scrub.detectors.encoder import EncoderDetector
from pii_scrub.detectors.presidio import PresidioDetector
from pii_scrub.detectors.regex import RegexDetector
from research.data.artifacts import load_dataset_split
from research.data.conll import load_conll2003_record
from research.data.models import DatasetExample
from research.data.ood_io import load_ood_jsonl
from research.eval.ablation import wrap_merge_ablation
from research.eval.encoder import load_encoder_predictor
from research.eval.qwen_detector import QwenDetector
from research.eval.qwen_inference import load_qwen_generator
from research.eval.report_io import write_report
from research.eval.run_baseline import evaluate_baseline

DETECTORS: dict[str, Callable[[], Detector]] = {
    "regex": RegexDetector,
    "presidio": PresidioDetector,
}


def main() -> None:
    """Parse arguments, evaluate one detector, and write JSON."""

    args = _parser().parse_args()

    detector = _build_detector(
        args.detector,
        args.model_path,
        args.device,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        merge_person_fragments=args.merge_person_fragments,
    )

    examples = _load_examples(
        args.dataset,
        args.input,
        args.split,
        args.data_file,
    )

    entities = {"PERSON"} if args.dataset == "conll" else None

    report = evaluate_baseline(
        detector,
        examples,
        entities=entities,
    )

    if isinstance(detector, QwenDetector):
        report["parse_failures"] = detector.parse_failures
        report["parse_failure_rate"] = detector.parse_failures / len(examples) if examples else 0.0

    write_report(
        report,
        args.output,
    )


def _build_detector(
    name: str,
    model_path: Path | None,
    device: str | None,
    *,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 256,
    merge_person_fragments: bool = False,
) -> Detector:
    """Build one configured detector for evaluation."""

    if name in DETECTORS:
        if merge_person_fragments:
            raise ValueError("--merge-person-fragments is supported only for encoder")

        return DETECTORS[name]()

    if name == "qwen":
        if merge_person_fragments:
            raise ValueError("--merge-person-fragments is supported only for encoder")

        if model_path is None:
            raise ValueError("--model-path is required for the qwen detector")

        generator = load_qwen_generator(
            model_path,
            base_model_name=base_model,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        return QwenDetector(generator)

    if model_path is None:
        raise ValueError("--model-path is required for the encoder detector")

    predictor = load_encoder_predictor(
        model_path,
        device=device,
    )

    if merge_person_fragments:
        predictor = wrap_merge_ablation(
            predictor,
            entity_types={"PERSON"},
        )

    return EncoderDetector(predictor)


def _load_examples(
    dataset: str,
    path: Path | None,
    split: str | None,
    data_file: str | None,
) -> Sequence[DatasetExample]:
    """Load normalized examples for one supported source."""

    if dataset == "ood":
        if path is None:
            raise ValueError("--input is required for OOD")

        return load_ood_jsonl(path)

    if dataset == "artifact":
        if path is None:
            raise ValueError("--input is required for artifact datasets")

        if split is None:
            raise ValueError("--split is required for artifact datasets")

        artifact = load_dataset_split(path)

        if split == "train":
            return artifact.split.train

        if split == "validation":
            return artifact.split.validation

        return artifact.split.test

    if data_file is None:
        raise ValueError("--data-file is required for CoNLL")

    rows = load_dataset(
        "parquet",
        data_files=data_file,
        split=split or "train",
    )

    return tuple(load_conll2003_record(row) for row in rows)


def _parser() -> ArgumentParser:
    """Build the detector evaluation CLI parser."""

    parser = ArgumentParser(
        description="Run one PII detector evaluation.",
    )

    parser.add_argument(
        "--detector",
        choices=(*DETECTORS, "encoder", "qwen"),
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        choices=("ood", "artifact", "conll"),
        default="ood",
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
    )

    parser.add_argument(
        "--data-file",
        type=str,
    )

    parser.add_argument(
        "--input",
        type=Path,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
    )

    parser.add_argument(
        "--device",
        type=str,
    )

    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--merge-person-fragments",
        action="store_true",
        help="Merge encoder spans separated only by non-alphanumeric text.",
    )

    return parser


if __name__ == "__main__":
    main()
