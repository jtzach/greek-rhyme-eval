"""Strict record validation implemented with the Python standard library."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

FEATURES = frozenset({"PURE", "RICH", "IMPERFECT", "MOSAIC", "IDV", "COPY"})
TYPES = frozenset({"M", "F2", "F3"})
ANNOTATION_STATUSES = frozenset(
    {"synthetic", "automatic", "human_single", "human_agreement", "human_adjudicated"}
)
HUMAN_GOLD_STATUSES = frozenset({"human_agreement", "human_adjudicated"})


class ValidationError(ValueError):
    """Invalid record, reference provenance, or prediction alignment."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _keys(obj: Any, required: set[str], optional: set[str], location: str) -> None:
    _require(isinstance(obj, dict), f"{location}: expected an object")
    _require(required <= obj.keys(), f"{location}: missing fields {sorted(required - obj.keys())}")
    _require(obj.keys() <= required | optional, f"{location}: unknown fields {sorted(obj.keys() - required - optional)}")


def _strings(value: Any, location: str) -> None:
    _require(isinstance(value, list), f"{location}: expected a list")
    _require(all(_text(item) for item in value), f"{location}: expected nonempty strings")
    _require(len(value) == len(set(value)), f"{location}: duplicates are forbidden")


def validate_label(label: Any, location: str = "label") -> None:
    _keys(label, {"rhyme", "rhyme_type", "features"}, set(), location)
    _require(label["rhyme"] in ("yes", "no", "uncertain"), f"{location}: invalid rhyme decision")
    _strings(label["features"], f"{location}.features")
    _require(set(label["features"]) <= FEATURES, f"{location}: use canonical feature names {sorted(FEATURES)}")
    if label["rhyme"] == "yes":
        _require(isinstance(label["rhyme_type"], str) and label["rhyme_type"] in TYPES, f"{location}: positive rhyme requires M, F2 or F3")
    else:
        _require(label["rhyme_type"] is None and label["features"] == [], f"{location}: no/uncertain requires null type and empty features")


def validate_dataset(records: Any) -> dict[str, Any]:
    _require(isinstance(records, list) and bool(records), "dataset: expected a nonempty list")
    seen: set[str] = set()
    poem_splits: dict[tuple[str, str], str] = {}
    for index, record in enumerate(records):
        loc = f"dataset[{index}]"
        _keys(record, {"pair_id", "language", "source_id", "poem_id", "split", "lines", "label", "annotation"}, {"notes"}, loc)
        for field in ("pair_id", "language", "source_id", "poem_id"):
            _require(_text(record[field]), f"{loc}.{field}: expected a nonempty string")
        _require(record["pair_id"] not in seen, f"{loc}: duplicate pair_id {record['pair_id']}")
        seen.add(record["pair_id"])
        _require(record["split"] in ("train", "dev", "test", "legacy"), f"{loc}: invalid split")
        _require(isinstance(record["lines"], list) and len(record["lines"]) == 2 and all(_text(line) for line in record["lines"]), f"{loc}: exactly two nonempty lines required")
        key = (record["source_id"], record["poem_id"])
        _require(key not in poem_splits or poem_splits[key] == record["split"], f"{loc}: poem occurs in multiple splits")
        poem_splits[key] = record["split"]
        validate_label(record["label"], f"{loc}.label")
        annotation = record["annotation"]
        _keys(annotation, {"status", "annotator_ids", "adjudicator_id"}, set(), f"{loc}.annotation")
        _require(isinstance(annotation["status"], str) and annotation["status"] in ANNOTATION_STATUSES, f"{loc}: invalid annotation status")
        _strings(annotation["annotator_ids"], f"{loc}.annotation.annotator_ids")
        _require(annotation["adjudicator_id"] is None or _text(annotation["adjudicator_id"]), f"{loc}: invalid adjudicator ID")
        count = len(annotation["annotator_ids"])
        status = annotation["status"]
        _require(status not in HUMAN_GOLD_STATUSES or count >= 2, f"{loc}: agreement/adjudication requires at least two distinct annotators")
        _require(status != "human_single" or count == 1, f"{loc}: human_single requires one annotator")
        _require(status not in ("automatic", "synthetic") or count == 0, f"{loc}: automatic/synthetic records must not claim human annotators")
        _require(status != "human_adjudicated" or _text(annotation["adjudicator_id"]), f"{loc}: adjudicated label needs adjudicator ID")
        _require(status == "human_adjudicated" or annotation["adjudicator_id"] is None, f"{loc}: adjudicator ID only applies to human_adjudicated")
        if "notes" in record:
            _require(isinstance(record["notes"], str), f"{loc}: notes must be a string")
    return {
        "items": len(records),
        "annotation_statuses": dict(Counter(r["annotation"]["status"] for r in records)),
        "rhyme_decisions": dict(Counter(r["label"]["rhyme"] for r in records)),
        "splits": dict(Counter(r["split"] for r in records)),
        "poems": len(poem_splits),
    }


def validate_predictions(records: Any, known_ids: set[str]) -> None:
    _require(isinstance(records, list), "predictions: expected a list")
    seen: set[str] = set()
    for index, record in enumerate(records):
        loc = f"predictions[{index}]"
        _keys(record, {"pair_id", "status", "rhyme", "rhyme_type", "features"}, {"error", "raw_output_path"}, loc)
        _require(_text(record["pair_id"]), f"{loc}: invalid pair_id")
        _require(record["pair_id"] in known_ids, f"{loc}: unknown pair_id {record['pair_id']}")
        _require(record["pair_id"] not in seen, f"{loc}: duplicate pair_id {record['pair_id']}")
        seen.add(record["pair_id"])
        _require(record["status"] in ("ok", "abstain", "error"), f"{loc}: invalid status")
        validate_label({key: record[key] for key in ("rhyme", "rhyme_type", "features")}, loc)
        _require(record["status"] == "ok" or record["rhyme"] == "uncertain", f"{loc}: abstain/error requires uncertain decision")
        _require(record["status"] != "error" or _text(record.get("error")), f"{loc}: error status requires an error message")
        _require(record["status"] == "error" or "error" not in record, f"{loc}: error message requires error status")
        if "raw_output_path" in record:
            _require(_text(record["raw_output_path"]), f"{loc}: invalid raw output path")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array or one JSON object per nonempty JSONL line."""
    path = Path(path)
    try:
        contents = path.read_text(encoding="utf-8-sig")
        records = ([json.loads(line) for line in contents.splitlines() if line.strip()]
                   if path.suffix.lower() == ".jsonl" else json.loads(contents))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: {exc}") from exc
    _require(isinstance(records, list), f"{path}: expected a JSON array or JSONL records")
    return records
