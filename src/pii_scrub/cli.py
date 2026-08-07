"""Command-line interface for local regex-based redaction."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pii_scrub.api import Scrubber


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without executing the command."""

    parser = argparse.ArgumentParser(prog="pii-scrub", description="Redact PII locally.")
    parser.add_argument("text", nargs="?", help="Text to redact.")
    parser.add_argument("--file", type=Path, help="Read UTF-8 text from a file.")
    parser.add_argument("--entities", help="Comma-separated entity types.")
    parser.add_argument("--mapping-out", type=Path, help="Write restoration mapping as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    if bool(args.text) == bool(args.file):
        raise SystemExit("provide exactly one of TEXT or --file")
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    entities = set(args.entities.split(",")) if args.entities else None
    result = Scrubber().scrub(text, entities=entities)
    print(result.text)
    if args.mapping_out:
        payload = [asdict(entry) for entry in result.mapping]
        args.mapping_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
