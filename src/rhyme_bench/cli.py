"""Offline CLI: validate a reference and score saved structured predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .contracts import ValidationError, load_records, validate_dataset
from .scoring import score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-dataset")
    validate.add_argument("dataset", type=Path)
    evaluate = subcommands.add_parser("score")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--allow-nonhuman-reference", action="store_true")
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        records = load_records(args.dataset)
        if args.command == "validate-dataset":
            result = {"valid": True, **validate_dataset(records)}
        else:
            result = score(records, load_records(args.predictions), allow_nonhuman_reference=args.allow_nonhuman_reference)
            result["inputs"] = {key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for key, path in (("dataset", args.dataset), ("predictions", args.predictions))}
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.command == "score" and args.output:
            output = args.output.resolve()
            if output in (args.dataset.resolve(), args.predictions.resolve()):
                raise ValidationError("Output path must not overwrite dataset or predictions")
            if output.exists():
                if any(output.samefile(path) for path in (args.dataset, args.predictions)):
                    raise ValidationError("Output path aliases dataset or predictions")
                raise ValidationError("Output already exists; choose a new report path")
            output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also prevents a race from clobbering an existing file.
            with output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        else:
            print(serialized, end="")
    except (ValidationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
