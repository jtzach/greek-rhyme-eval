"""Unambiguous metric denominators; ambiguous references are not forced to gold."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .contracts import HUMAN_GOLD_STATUSES, ValidationError, validate_dataset, validate_predictions


def _fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"correct": numerator, "total": denominator, "value": numerator / denominator if denominator else None}


def _prf(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def score(dataset: list[dict[str, Any]], predictions: list[dict[str, Any]], *, allow_nonhuman_reference: bool = False) -> dict[str, Any]:
    """Score all definite references; missing outputs stay in the denominator."""
    inventory = validate_dataset(dataset)
    validate_predictions(predictions, {item["pair_id"] for item in dataset})
    unverified = sum(item["annotation"]["status"] not in HUMAN_GOLD_STATUSES for item in dataset)
    if unverified and not allow_nonhuman_reference:
        raise ValidationError("Reference includes synthetic, automatic or single-annotator labels; default scoring requires human agreement/adjudication. Use --allow-nonhuman-reference only for explicit diagnostic/plumbing evaluation.")
    pred_by_id = {prediction["pair_id"]: prediction for prediction in predictions}
    definite = [item for item in dataset if item["label"]["rhyme"] != "uncertain"]
    positive = sum(item["label"]["rhyme"] == "yes" for item in definite)
    detection_correct = type_correct = joint_correct = feature_exact = answered = 0
    tp = fp = fn = feature_tp = feature_fp = feature_fn = 0
    statuses: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    item_results = []
    for item in definite:
        gold = item["label"]
        prediction = pred_by_id.get(item["pair_id"])
        outcome = ("missing" if prediction is None else prediction["status"])
        if prediction is not None and outcome == "ok" and prediction["rhyme"] == "uncertain":
            outcome = "abstain"
        statuses[outcome] += 1
        usable = outcome == "ok"
        predicted_rhyme = prediction["rhyme"] if usable else outcome
        confusion[f"{gold['rhyme']}->{predicted_rhyme}"] += 1
        answered += usable
        detected = usable and predicted_rhyme == gold["rhyme"]
        detection_correct += detected
        tp += gold["rhyme"] == "yes" and predicted_rhyme == "yes"
        fp += gold["rhyme"] == "no" and predicted_rhyme == "yes"
        fn += gold["rhyme"] == "yes" and predicted_rhyme != "yes"
        type_match: bool | None = None
        feature_match: bool | None = None
        if gold["rhyme"] == "yes":
            # Null never matches a type. Missed detection/error is also a type error.
            type_match = bool(usable and predicted_rhyme == "yes" and prediction["rhyme_type"] == gold["rhyme_type"])
            type_correct += type_match
            gold_features = set(gold["features"])
            predicted_features = set(prediction["features"]) if usable and predicted_rhyme == "yes" else set()
            feature_tp += len(gold_features & predicted_features)
            feature_fp += len(predicted_features - gold_features)
            feature_fn += len(gold_features - predicted_features)
            feature_match = bool(usable and predicted_rhyme == "yes" and gold_features == predicted_features)
            feature_exact += feature_match
            joint = type_match and feature_match
        else:
            # No-rhyme exact match is a decision only, never a type true positive.
            joint = bool(detected)
        joint_correct += joint
        item_results.append({"pair_id": item["pair_id"], "prediction_status": outcome,
                             "detection_correct": bool(detected), "type_correct": type_match,
                             "feature_exact": feature_match, "joint_correct": bool(joint)})
    return {
        "schema_version": "0.1.0",
        "reference": {**inventory, "nonconsensus_reference_items": unverified,
                      "warning": "DIAGNOSTIC ONLY: includes non-consensus/nonhuman labels; not a human-gold benchmark score." if unverified else None,
                      "provenance_note": "Human status is declared metadata, not externally certified by this software."},
        "counts": {"reference_items": len(dataset), "scored_definite_items": len(definite),
                   "excluded_uncertain_reference_items": len(dataset) - len(definite),
                   "positive_reference_items": positive, "provided_predictions": len(predictions),
                   "prediction_statuses_on_definite_items": dict(statuses)},
        "metrics": {"detection_accuracy": _fraction(detection_correct, len(definite)),
                    "detection_positive_class": _prf(tp, fp, fn),
                    "answer_coverage": _fraction(answered, len(definite)),
                    "type_accuracy_on_gold_positives": _fraction(type_correct, positive),
                    "feature_micro_on_gold_positives": _prf(feature_tp, feature_fp, feature_fn),
                    "feature_exact_on_gold_positives": _fraction(feature_exact, positive),
                    "joint_accuracy_on_definite_items": _fraction(joint_correct, len(definite))},
        "detection_confusion": dict(confusion),
        "excluded_uncertain_pair_ids": [item["pair_id"] for item in dataset if item["label"]["rhyme"] == "uncertain"],
        "item_results": item_results,
    }
