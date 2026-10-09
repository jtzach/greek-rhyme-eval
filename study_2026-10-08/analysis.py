"""Offline analysis of the frozen Greek automatic-reference study.

No model calls, label correction, or output repair. Component feature metrics
score the returned set even when its type is NONE; joint agreement also requires
the reference stress type. Failed batches remain in all planned denominators.
Run only after both phases finish (or explicitly request --allow-partial).
Figures require numpy/matplotlib, available in the bundled Python runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
FEATURES = ("PURE", "RICH", "IMPERFECT", "IDV", "MOSAIC", "COPY")
CORE_EXCLUDED = frozenset({"MOSAIC", "COPY"})
CONDITIONS = ("S", "C", "E", "R", "V", "W")
CONTRASTS = (("C", "S"), ("E", "S"), ("R", "S"), ("V", "R"), ("W", "V"))
METRICS = ("joint", "stress", "feature_exact", "feature_item_f1", "core_joint")
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "reasoning_output_tokens")
MODEL_LABELS = {"astra": "Astra", "sol": "Sol", "luna": "Luna"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_prediction(reference, prediction=None, valid=True):
    """All references are positive. Invalid outputs predict no features for FN.

    Do not normalize or infer any features here: the input contracts are frozen.
    Empty/empty F1 is one only for a valid prediction, following the protocol.
    """
    true_type = reference["position"]
    true_features = set(reference["features"])
    valid = bool(valid and prediction is not None)
    predicted_type = prediction.get("type") if valid else None
    predicted_features = set(prediction["features"]) if valid else set()
    tp = len(true_features & predicted_features)
    fp = len(predicted_features - true_features)
    fn = len(true_features - predicted_features)
    stress = valid and predicted_type == true_type
    exact = valid and predicted_features == true_features
    item_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float(valid)
    return {"valid": int(valid), "stress": int(stress), "feature_exact": int(exact),
            "joint": int(stress and exact), "feature_item_f1": item_f1,
            "core_joint": int(stress and (predicted_features - CORE_EXCLUDED) == (true_features - CORE_EXCLUDED)),
            "feature_tp": tp, "feature_fp": fp, "feature_fn": fn,
            "none_with_nonempty_features": int(valid and predicted_type == "NONE" and bool(predicted_features)),
            "predicted_none": int(valid and predicted_type == "NONE")}


def summarize_metrics(rows):
    if not rows:
        return {"n": 0}
    result = {"n": len(rows)}
    for metric in METRICS + ("valid", "none_with_nonempty_features", "predicted_none"):
        result[metric] = sum(row[metric] for row in rows) / len(rows)
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("feature_tp", "feature_fp", "feature_fn"))
    result.update(feature_tp=tp, feature_fp=fp, feature_fn=fn,
                  feature_micro_precision=ratio(tp, tp + fp),
                  feature_micro_recall=ratio(tp, tp + fn),
                  feature_micro_f1=ratio(2 * tp, 2 * tp + fp + fn))
    return result


def mean_replicates(rows, expected_replicates=2):
    """Produce one paired analysis record per item/model/condition."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["model"], row["condition"], row["id"])].append(row)
    output = []
    keys = METRICS + ("valid", "feature_tp", "feature_fp", "feature_fn", "none_with_nonempty_features", "predicted_none")
    for (split, model, condition, item_id), group in sorted(grouped.items()):
        group.sort(key=lambda row: row["replicate"])
        if len(group) != expected_replicates or {r["replicate"] for r in group} != set(range(1, expected_replicates + 1)):
            raise ValueError(f"Missing/duplicated planned replicate for {item_id}/{model}/{condition}")
        first = group[0]
        if any(row["batch"] != first["batch"] or row["group"] != first["group"] for row in group):
            raise ValueError("Replicate grouping changed")
        complete_valid = all(row["valid"] for row in group)
        signatures = [(row["predicted_type"], tuple(sorted(row["predicted_features"]))) for row in group]
        record = {"split": split, "model": model, "condition": condition, "id": item_id,
                  "batch": first["batch"], "group": first["group"], "reference_type": first["reference_type"],
                  "reference_features": first["reference_features"], "replicates": expected_replicates,
                  "both_replicates_valid": int(complete_valid),
                  "stable_valid_label_set": int(complete_valid and len(set(signatures)) == 1),
                  "replicate_predictions": [dict(replicate=row["replicate"], valid=bool(row["valid"]),
                                                  type=row["predicted_type"], features=row["predicted_features"]) for row in group]}
        record.update({key: sum(row[key] for row in group) / expected_replicates for key in keys})
        output.append(record)
    return output


def cluster_bootstrap(values, groups, resamples=10000, seed=20261008):
    """Percentile CI for an item-weighted mean, resampling whole clusters.

    Item outcomes must already average repetitions. Unequal author groups are
    weighted by their resampled item counts, not treated as equal-sized authors.
    The same seed/group order couples draws across conditions and contrasts.
    """
    import numpy as np
    if len(values) != len(groups) or not values:
        raise ValueError("Nonempty aligned values/groups required")
    unique = sorted(set(groups), key=str)
    indices = {group: index for index, group in enumerate(unique)}
    sums = np.zeros(len(unique), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    for value, group in zip(values, groups):
        sums[indices[group]] += float(value)
        counts[indices[group]] += 1
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(unique), size=(resamples, len(unique)))
    means = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"estimate": float(np.mean(values)), "lower": float(lo), "upper": float(hi),
            "clusters": len(unique), "items": len(values), "resamples": resamples, "seed": seed,
            "method": "item-weighted paired cluster percentile bootstrap"}


def paired_contrast(a, b, group_key="batch", resamples=10000, seed=20261008, metric="joint"):
    by_id_a = {row["id"]: row for row in a}
    by_id_b = {row["id"]: row for row in b}
    if len(by_id_a) != len(a) or len(by_id_b) != len(b) or by_id_a.keys() != by_id_b.keys():
        raise ValueError("Paired contrasts require identical unique item IDs")
    ids = sorted(by_id_a)
    for item_id in ids:
        if by_id_a[item_id][group_key] != by_id_b[item_id][group_key]:
            raise ValueError("Paired cluster labels differ")
    return cluster_bootstrap([by_id_a[i][metric] - by_id_b[i][metric] for i in ids],
                             [by_id_a[i][group_key] for i in ids], resamples, seed)


def _load_runner(root):
    spec = importlib.util.spec_from_file_location("frozen_study_runner", root / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_freezes(root):
    protocol = read(root / "protocol_frozen.json")
    for relative, expected in protocol["hashes"].items():
        if sha(root / relative) != expected:
            raise ValueError("Frozen protocol artifact changed: " + relative)
    for relative, expected in read(root / "data/freeze.json")["files_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError("Frozen dataset artifact changed: " + relative)
    return protocol


def planned_jobs(root, models):
    phase1 = read(root / "manifests/phase1.json")["jobs"]
    revision = read(root / "design/revision_plan.json")
    phase2_rows = read(root / "manifests/phase2.json")["jobs"] if (root / "manifests/phase2.json").exists() else []
    blocked_rows = read(root / "manifests/phase2_blocked.json") if (root / "manifests/phase2_blocked.json").exists() else []
    phase2 = {j["id"]: j for j in phase2_rows}
    blocked = {j["id"]: j for j in blocked_rows}
    planned_ids = {j["id"] for j in revision}
    if len(phase2) != len(phase2_rows) or len(blocked) != len(blocked_rows):
        raise ValueError("Duplicate phase2 or blocked job IDs")
    if (set(phase2) | set(blocked)) - planned_ids or set(phase2) & set(blocked):
        raise ValueError("Unexpected/overlapping phase2 or blocked job IDs")
    if (root / "manifests/phase2.json").exists() and set(phase2) | set(blocked) != planned_ids:
        raise ValueError("Phase2 and blocked manifests do not partition the frozen revision plan")
    result = [{**job, "phase": "phase1", "blocked": False} for job in phase1]
    for planned in revision:
        job = dict(planned)
        job["item_ids"] = [item["id"] for item in planned["items"]]
        supplied = phase2.get(planned["id"], blocked.get(planned["id"]))
        if supplied is not None:
            for key in ("id", "model_key", "condition", "replicate", "batch", "split", "dependency"):
                if supplied.get(key) != planned[key]:
                    raise ValueError("Phase2 changed planned metadata: " + planned["id"] + "/" + key)
        if planned["id"] in phase2:
            if phase2[planned["id"]]["item_ids"] != job["item_ids"]:
                raise ValueError("Phase2 changed planned item IDs")
            if phase2[planned["id"]].get("model") != models[planned["model_key"]]:
                raise ValueError("Phase2 changed requested model")
            job.update(phase2[planned["id"]])
        job.update(phase="phase2", blocked=planned["id"] in blocked)
        result.append(job)
    if len({job["id"] for job in result}) != len(result):
        raise ValueError("Duplicate planned job ID")
    return result


def cost_record(folder, meta):
    """Count all committed attempts, including failed transport retries."""
    attempts = []
    for directory in sorted(folder.glob("attempt_*")):
        path = directory / "meta.json"
        if path.exists():
            attempts.append(read(path))
        elif (directory / "recovery_failure.json").exists():
            attempts.append(read(directory / "recovery_failure.json"))
    if not attempts and meta is not None:
        attempts = [meta]
    known_seconds = [attempt.get("elapsed_seconds") for attempt in attempts
                     if attempt.get("elapsed_seconds") is not None and not attempt.get("interrupted_unknown_outcome")]
    result = {"attempts": len(attempts), "attempts_with_usage": sum(isinstance(a.get("usage"), dict) for a in attempts),
              "all_attempt_seconds": sum(known_seconds) if known_seconds else None,
              "final_attempt_seconds": meta.get("elapsed_seconds") if meta and not meta.get("interrupted_unknown_outcome") else None,
              "incomplete_latency": len(known_seconds) != len(attempts),
              "incomplete_usage": any(not isinstance(a.get("usage"), dict) for a in attempts)}
    for key in USAGE_KEYS:
        values = [a["usage"][key] for a in attempts if isinstance(a.get("usage"), dict) and a["usage"].get(key) is not None]
        result[key] = sum(values) if values else None
    return result


def collect_predictions(root, protocol, allow_partial=False):
    runner = _load_runner(root)
    refs = {}
    for split in ("main300", "legacy40"):
        refs[split] = {r["id"]: r for r in read(root / "data/references" / (split + ".json"))}
    records, jobs, absent = [], [], []
    configs = {phase: read(root / "runs" / phase / "run_config.json") if (root / "runs" / phase / "run_config.json").exists() else None
               for phase in ("phase1", "phase2")}
    for phase, config in configs.items():
        if config is None:
            continue
        if config.get("runner_sha256") != protocol["hashes"]["runner.py"]:
            raise ValueError("Run used a different runner: " + phase)
        if config.get("reasoning_effort") != protocol["reasoning_effort"]:
            raise ValueError("Run reasoning effort differs: " + phase)
        if config.get("manifest_sha256") != sha(root / "manifests" / (phase + ".json")):
            raise ValueError("Run manifest changed: " + phase)
        if config.get("wrapper_sha256") != hashlib.sha256(runner.WRAPPER.encode()).hexdigest():
            raise ValueError("Run wrapper differs: " + phase)
    for job in planned_jobs(root, protocol["models"]):
        folder = root / "runs" / job["phase"] / job["id"]
        meta_path = folder / "meta.json"
        meta = read(meta_path) if meta_path.exists() else None
        status = "blocked_by_initial_failure" if job["blocked"] else "missing_final_metadata"
        predictions = {}
        if job["phase"] == "phase2" and (root / "manifests/phase2.json").exists():
            dependency_folder = root / "runs/phase1" / job["dependency"]
            dependency_meta_path = dependency_folder / "meta.json"
            if not dependency_meta_path.exists():
                raise ValueError("Phase2 was constructed without committed S metadata: " + job["id"])
            dependency_meta = read(dependency_meta_path)
            dependency_response = (dependency_folder / "response.txt").read_text(encoding="utf-8")
            dependency_hash = hashlib.sha256(dependency_response.encode()).hexdigest()
            if dependency_hash != dependency_meta.get("response_sha256"):
                raise ValueError("Shared S response changed: " + job["id"])
            if job["blocked"] != (not bool(dependency_meta.get("valid"))):
                raise ValueError("Blocked revision status contradicts initial S validity: " + job["id"])
            if not job["blocked"] and job.get("dependency_response_sha256") != dependency_hash:
                raise ValueError("Revision did not record the frozen shared S response hash: " + job["id"])
            if job["blocked"] and meta is not None:
                raise ValueError("Blocked revision unexpectedly has inference output: " + job["id"])
        if meta is not None:
            if meta.get("id") != job["id"]:
                raise ValueError("Saved metadata job ID differs: " + job["id"])
            response_path = folder / "response.txt"
            response = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
            if hashlib.sha256(response.encode("utf-8")).hexdigest() != meta.get("response_sha256"):
                raise ValueError("Response hash differs: " + job["id"])
            config = configs[job["phase"]]
            if not config:
                raise ValueError("Missing run configuration: " + job["phase"])
            if meta.get("model_requested") != protocol["models"][job["model_key"]] or meta.get("reasoning_effort_requested") != protocol["reasoning_effort"]:
                raise ValueError("Model/effort mismatch: " + job["id"])
            prompt_path = (root / "manifests" / job["prompt_file"]).resolve()
            if sha(prompt_path) != config["prompt_hashes"][job["id"]]:
                raise ValueError("Prompt hash mismatch: " + job["id"])
            prompt = runner.WRAPPER + prompt_path.read_text(encoding="utf-8")
            if hashlib.sha256(prompt.encode()).hexdigest() != meta.get("prompt_sha256"):
                raise ValueError("Sent-prompt hash mismatch: " + job["id"])
            if (folder / "input.txt").read_text(encoding="utf-8") != prompt:
                raise ValueError("Saved sent input differs: " + job["id"])
            format_ok, _ = runner.validate_answer(response, job["item_ids"])
            inference_ok = meta.get("returncode") == 0 and meta.get("completed") is True and meta.get("tool_event_count") == 0
            valid = bool(meta.get("valid") and inference_ok and format_ok)
            if valid:
                predictions = {p["id"]: p for p in json.loads(response)["predictions"]}
                status = "valid"
            elif meta.get("tool_event_count"):
                status = "tool_use"
            elif not inference_ok:
                status = "transport_or_interrupted_failure"
            else:
                status = "invalid_format"
        elif not job["blocked"]:
            absent.append(job["id"])
        costs = cost_record(folder, meta)
        job_record = {"job_id": job["id"], "phase": job["phase"], "split": job["split"],
                      "model": job["model_key"], "model_requested": protocol["models"][job["model_key"]],
                      "condition": job["condition"], "replicate": job["replicate"], "batch": job["batch"],
                      "items": len(job["item_ids"]), "status": status, "dependency": job.get("dependency"), **costs}
        jobs.append(job_record)
        for item_id in job["item_ids"]:
            source = refs[job["split"]][item_id]
            ref = source["reference"]
            prediction = predictions.get(item_id)
            row = {"id": item_id, "split": job["split"], "model": job["model_key"],
                   "model_requested": protocol["models"][job["model_key"]], "condition": job["condition"],
                   "replicate": job["replicate"], "batch": job["batch"],
                   "group": source["provenance"].get("grouping_key") or "UNRESOLVED_SOURCE_GROUP",
                   "legacy_subset": source["provenance"].get("legacy_subset"),
                   "job_id": job["id"], "status": status,
                   "reference_type": ref["position"], "reference_features": sorted(ref["features"]),
                   "predicted_type": prediction["type"] if prediction else None,
                   "predicted_features": sorted(prediction["features"]) if prediction else [],
                   "response_sha256": meta.get("response_sha256") if meta else None}
            row.update(score_prediction(ref, prediction, valid=status == "valid"))
            records.append(row)
    expected = protocol["counts"]["main_prediction_records"] + protocol["counts"]["legacy_prediction_records"]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} planned prediction records, found {len(records)}")
    if absent and not allow_partial:
        raise RuntimeError(f"{len(absent)} batches have no final metadata; analysis requires complete phases or --allow-partial")
    if not (root / "manifests/phase2.json").exists() and not allow_partial:
        raise RuntimeError("Phase2 manifest not created")
    return records, jobs, absent


def summarize_cost(jobs):
    result = {"planned_jobs": len(jobs), "committed_attempts": sum(j["attempts"] for j in jobs),
              "jobs_with_attempts": sum(j["attempts"] > 0 for j in jobs),
              "jobs_without_usage": sum(j["attempts"] > 0 and not j["attempts_with_usage"] for j in jobs),
              "jobs_with_incomplete_attempt_usage": sum(j["incomplete_usage"] for j in jobs),
              "attempts_without_usage": sum(j["attempts"] - j["attempts_with_usage"] for j in jobs),
              "jobs_with_incomplete_latency": sum(j["incomplete_latency"] for j in jobs),
              "status_counts": dict(Counter(j["status"] for j in jobs))}
    for key in USAGE_KEYS + ("all_attempt_seconds",):
        values = [j[key] for j in jobs if j[key] is not None]
        result[key] = sum(values) if values else None
        result[key + "_jobs_reporting"] = len(values)
    latencies = [j["all_attempt_seconds"] for j in jobs if j["all_attempt_seconds"] is not None and not j["incomplete_latency"]]
    result["median_call_batch_seconds"] = statistics.median(latencies) if latencies else None
    result["mean_call_batch_seconds"] = statistics.mean(latencies) if latencies else None
    return result


def pipeline_latency(own_jobs, all_jobs):
    """Sum shared S and revision latency per batch, rather than averaging stages."""
    by_id = {job["job_id"]: job for job in all_jobs}
    values = []
    for job in own_jobs:
        stages = [job]
        if job.get("dependency"):
            stages.append(by_id[job["dependency"]])
        if all(stage["all_attempt_seconds"] is not None and not stage["incomplete_latency"] for stage in stages):
            values.append(sum(stage["all_attempt_seconds"] for stage in stages))
    return {"planned_batches": len(own_jobs), "known_elapsed_batches": len(values),
            "unknown_elapsed_batches": len(own_jobs) - len(values),
            "median_end_to_end_batch_seconds": statistics.median(values) if values else None,
            "mean_end_to_end_batch_seconds": statistics.mean(values) if values else None,
            "note": "Sum of recorded inference-attempt durations across stages, excluding queue/wait time; not experiment wall-clock duration"}


def baseline_records(root):
    analyses = read(root / "design/verifier_analyses.json")
    output = []
    for split in ("main300", "legacy40"):
        batches = {item["id"]: index // 10 + 1
                   for index, item in enumerate(read(root / "data/inputs" / (split + ".json")))}
        for index, reference in enumerate(read(root / "data/references" / (split + ".json"))):
            for version in ("original", "repaired"):
                value = analyses[version][reference["id"]]
                prediction = {"type": value["stress_type"], "features": value["features"]}
                row = {"id": reference["id"], "split": split, "version": version,
                       "batch": batches[reference["id"]],
                       "group": reference["provenance"].get("grouping_key") or "UNRESOLVED_SOURCE_GROUP",
                       "reference_type": reference["reference"]["position"],
                       "reference_features": sorted(reference["reference"]["features"]),
                       "predicted_type": value["stress_type"], "predicted_features": sorted(value["features"]),
                       "engine_accepts_rhyme": bool(value["valid"])}
                row.update(score_prediction(reference["reference"], prediction,
                                            valid=value["stress_type"] in {"M", "F2", "F3", "NONE"}))
                output.append(row)
    return output


def build_summary(records, items, jobs, baselines, protocol, resamples=10000, seed=20261008):
    summary = {"analysis_status": "final_if_all_planned_jobs_committed",
               "reference_status": "automatic stored corpus labels; agreement, not human-validated accuracy",
               "counts": {"prediction_records": len(records), "item_condition_records": len(items),
                          "main_prediction_records": sum(r["split"] == "main300" for r in records),
                          "legacy_prediction_records": sum(r["split"] == "legacy40" for r in records)},
               "statistics": {"resamples": resamples, "seed": seed,
                              "repetition_policy": "average two outcomes within item before cluster resampling",
                              "intervals": "descriptive paired 95% percentile intervals, without multiplicity adjustment; no significance declarations",
                              "grouping_caveat": "fixed inference batches for primary intervals; observed author/source groups for sensitivity, not known poem groups"},
               "metric_notes": {"feature_components": "Feature scores compare sets irrespective of type; joint also requires correct stress. Invalid inference contributes every reference feature as FN.",
                                "empty_sets": "Valid empty/empty per-item F1=1; invalid output F1=0. Undefined aggregate micro F1 is null.",
                                "core_joint": "Remove COPY and MOSAIC from both sets before exact joint comparison; no other normalization.",
                                "stability": "Both repetitions must be valid and have identical type/set; invalid-invalid is never counted stable. E uses a different frozen example draw in each repetition, so E stability combines demonstration-selection and decoding variability.",
                                "cost": "All committed attempts included. Condition R/V/W incremental costs exclude the reused S call; standalone-condition totals add its full recorded cost. Grand totals count physical attempts once. Latencies are per batch, not per-pair measured latency. Cached/reasoning token fields are separate counters, not added to their enclosing totals. No inferred dollars."},
               "cells": [], "contrasts": [], "baselines": [], "cost_total": summarize_cost(jobs)}
    for split in ("main300", "legacy40"):
        for model in protocol["models"]:
            for condition in (CONDITIONS if split == "main300" else ("S",)):
                subset = [r for r in items if (r["split"], r["model"], r["condition"]) == (split, model, condition)]
                predictions = [r for r in records if (r["split"], r["model"], r["condition"]) == (split, model, condition)]
                if not subset:
                    continue
                cell = {"split": split, "model": model, "model_requested": protocol["models"][model],
                        "condition": condition, "n_items": len(subset), "n_predictions": len(predictions),
                        "metrics": summarize_metrics(predictions), "by_position": {}, "intervals": {},
                        "stability": {"both_valid": sum(r["both_replicates_valid"] for r in subset),
                                      "stable_valid": sum(r["stable_valid_label_set"] for r in subset), "denominator": len(subset),
                                      "unconditional_rate": sum(r["stable_valid_label_set"] for r in subset) / len(subset),
                                      "conditional_on_both_valid": ratio(sum(r["stable_valid_label_set"] for r in subset), sum(r["both_replicates_valid"] for r in subset))}}
                for position in ("M", "F2", "F3"):
                    cell["by_position"][position] = summarize_metrics([r for r in predictions if r["reference_type"] == position])
                for metric in ("joint", "stress", "feature_exact", "core_joint"):
                    cell["intervals"][metric] = cluster_bootstrap([r[metric] for r in subset], [r["batch"] for r in subset], resamples, seed)
                cell["author_group_joint_sensitivity"] = cluster_bootstrap([r["joint"] for r in subset], [r["group"] for r in subset], resamples, seed)
                own_jobs = [j for j in jobs if (j["split"], j["model"], j["condition"]) == (split, model, condition)]
                cell["cost_incremental"] = summarize_cost(own_jobs)
                standalone_jobs = own_jobs + ([j for j in jobs if (j["split"], j["model"], j["condition"]) == (split, model, "S")] if condition in ("R", "V", "W") else [])
                cell["cost_standalone_condition"] = summarize_cost(standalone_jobs)
                cell["standalone_pipeline_latency"] = pipeline_latency(own_jobs, jobs)
                summary["cells"].append(cell)
    for model in protocol["models"]:
        for a, b in CONTRASTS:
            lhs = [r for r in items if r["split"] == "main300" and r["model"] == model and r["condition"] == a]
            rhs = [r for r in items if r["split"] == "main300" and r["model"] == model and r["condition"] == b]
            if not lhs or not rhs:
                continue
            summary["contrasts"].append({"model": model, "contrast": a + "-" + b,
                 "joint_batch": paired_contrast(lhs, rhs, "batch", resamples, seed),
                 "joint_author_group_sensitivity": paired_contrast(lhs, rhs, "group", resamples, seed),
                 "core_joint_batch": paired_contrast(lhs, rhs, "batch", resamples, seed, "core_joint")})
    for split in ("main300", "legacy40"):
        for version in ("original", "repaired"):
            subset = [r for r in baselines if r["split"] == split and r["version"] == version]
            summary["baselines"].append({"split": split, "version": version, "metrics": summarize_metrics(subset),
                 "intervals": {m: cluster_bootstrap([r[m] for r in subset], [r["batch"] for r in subset], resamples, seed) for m in ("joint", "core_joint")},
                 "engine_acceptance_rate": sum(r["engine_accepts_rhyme"] for r in subset) / len(subset)})
    original = {r["id"]: r for r in baselines if r["split"] == "main300" and r["version"] == "original"}
    repaired = {r["id"]: r for r in baselines if r["split"] == "main300" and r["version"] == "repaired"}
    summary["baseline_changes"] = {"n": len(original),
        "changed_type": sum(original[i]["predicted_type"] != repaired[i]["predicted_type"] for i in original),
        "changed_feature_set": sum(original[i]["predicted_features"] != repaired[i]["predicted_features"] for i in original),
        "joint_disagreement_to_agreement": sum(not original[i]["joint"] and repaired[i]["joint"] for i in original),
        "joint_agreement_to_disagreement": sum(original[i]["joint"] and not repaired[i]["joint"] for i in original)}
    return summary


def historical_diagnostic(root):
    path = root.parent / "evaluation/legacy_astra_2026-10-08/results/summary.json"
    if not path.exists():
        return {"status": "unavailable", "pooled_with_main_or_legacy40": False}
    value = read(path)
    return {"status": "archived separate diagnostic", "pooled_with_main_or_legacy40": False,
            "source_path": str(path), "sha256": sha(path), "n": value["n"],
            "model_requested": value["model_requested"], "metrics": value["metrics"],
            "incompatibilities": "26 records, literal historical CoT prompt/parser, one repetition; do not pool with structured study outcomes or compare as a controlled prompt effect"}


def latex_export(summary, path):
    def pct(value):
        return "--" if value is None else f"{100 * value:.1f}"
    lines = ["% Generated offline; automatic-reference agreement, not human-gold accuracy.",
             "% Intervals are descriptive 95% cluster-bootstrap intervals; no significance claims.",
             r"\begin{table*}[t]", r"\centering\small",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             "Model & Condition & Joint & Stress & Feature exact & Micro F1 & Core joint & Valid " + chr(92) * 2, r"\midrule"]
    for cell in summary["cells"]:
        if cell["split"] != "main300":
            continue
        m = cell["metrics"]
        values = [pct(m[k]) for k in ("joint", "stress", "feature_exact", "feature_micro_f1", "core_joint", "valid")]
        lines.append(" & ".join([MODEL_LABELS[cell["model"]], cell["condition"]] + values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\caption{Agreement with the stored automatic references on 300 Greek pairs, averaged over two repetitions. Values are percentages. Core joint excludes COPY and MOSAIC on both sides.}",
                  r"\label{tab:study-main-results}", r"\end{table*}", ""])
    lines.extend([r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{llrr}", r"\toprule",
                  "Model & Contrast & Gain (pp) & 95" + chr(92) + "% interval " + chr(92) * 2, r"\midrule"])
    for row in summary["contrasts"]:
        v = row["joint_batch"]
        lines.append(f"{MODEL_LABELS[row['model']]} & {row['contrast']} & {100*v['estimate']:+.1f} & [{100*v['lower']:+.1f}, {100*v['upper']:+.1f}]" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Descriptive paired differences in joint agreement; 10,000 resamples of the 30 fixed inference batches. No multiplicity-adjusted significance claim is made.}", r"\label{tab:study-contrasts}", r"\end{table}", ""])
    macros = ["% Generated percentage values; automatic-reference agreement."]
    for cell in summary["cells"]:
        prefix = "Study" + MODEL_LABELS[cell["model"]] + cell["condition"] + ("Legacy" if cell["split"] == "legacy40" else "")
        for key, suffix in (("joint", "Joint"), ("stress", "Stress"), ("feature_exact", "FeatureExact"), ("core_joint", "CoreJoint")):
            macros.append("\\providecommand{\\" + prefix + suffix + "}{" + pct(cell["metrics"][key]) + "}")
    lines.extend(macros)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(path).with_name("paper_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def create_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
    colors = {"astra": "#0072B2", "sol": "#D55E00", "luna": "#009E73"}
    models = list(MODEL_LABELS)
    cells = {(c["model"], c["condition"]): c for c in summary["cells"] if c["split"] == "main300"}

    def save(fig, name):
        if str(summary.get("analysis_status", "")).startswith("SYNTHETIC"):
            fig.suptitle("SYNTHETIC VALIDATION ONLY — NOT EXPERIMENTAL RESULTS", y=1.10, fontsize=10, color="#9B2226")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.7), layout="constrained")
    x = np.arange(len(CONDITIONS))
    for index, model in enumerate(models):
        ci = [cells[model, c]["intervals"]["joint"] for c in CONDITIONS]
        values = np.array([v["estimate"] * 100 for v in ci])
        errors = np.array([[max(0, values[i] - v["lower"] * 100) for i, v in enumerate(ci)],
                           [max(0, v["upper"] * 100 - values[i]) for i, v in enumerate(ci)]])
        ax.bar(x + (index - 1) * .24, values, .23, yerr=errors, capsize=3, color=colors[model], label=MODEL_LABELS[model])
    ax.set(xticks=x, xticklabels=CONDITIONS, ylabel="Joint agreement with automatic reference (%)", ylim=(0, 105),
           title="Main study: fixed 300 pairs, two repetitions")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.18)
    save(fig, "main_joint_comparison")

    contrasts = {(r["model"], r["contrast"]): r["joint_batch"] for r in summary["contrasts"]}
    names = [a + "-" + b for a, b in CONTRASTS]
    gain = np.array([[100 * contrasts[m, c]["estimate"] for c in names] for m in models])
    limit = max(5., float(np.max(np.abs(gain))))
    fig, ax = plt.subplots(figsize=(10.0, 3.3), layout="constrained")
    im = ax.imshow(gain, cmap="PuOr", vmin=-limit, vmax=limit, aspect="auto")
    for mi, model in enumerate(models):
        for ci, contrast in enumerate(names):
            value = contrasts[model, contrast]
            text_color = "white" if abs(gain[mi, ci]) > .66 * limit else "black"
            ax.text(ci, mi, f"{gain[mi,ci]:+.1f}\n[{100*value['lower']:+.1f}, {100*value['upper']:+.1f}]", ha="center", va="center", color=text_color)
    ax.set(xticks=range(len(names)), xticklabels=names, yticks=range(len(models)),
           yticklabels=[MODEL_LABELS[m] for m in models], title="Paired joint-agreement gains (percentage points)\nDescriptive 95% batch-bootstrap intervals; not significance tests")
    fig.colorbar(im, ax=ax, label="Gain (pp)", shrink=.8)
    save(fig, "paired_gains_intervals")

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.4), sharey=True, layout="constrained")
    for ax, model in zip(axes, models):
        for index, condition in enumerate(CONDITIONS):
            cell = cells[model, condition]["metrics"]
            stress, features = 100 * cell["stress"], 100 * cell["feature_exact"]
            ax.plot([features, stress], [index, index], color="#999999", linewidth=1.5, zorder=1)
            ax.scatter(stress, index, color="#0072B2", s=48, marker="o", zorder=2, label="Stress" if index == 0 else None)
            ax.scatter(features, index, color="#D55E00", s=50, marker="s", zorder=2, label="Feature exact" if index == 0 else None)
        ax.set(yticks=range(len(CONDITIONS)), yticklabels=CONDITIONS, xlim=(0, 103), ylim=(6.7, -.6),
               title=MODEL_LABELS[model])
        ax.grid(axis="x", alpha=.15)
    fig.supxlabel("Agreement with automatic reference (%)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    save(fig, "stress_vs_feature_agreement")

    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    dimensions = ("stress", "feature_exact", "joint", "core_joint")
    x = np.arange(len(dimensions))
    for index, version in enumerate(("original", "repaired")):
        row = next(r for r in summary["baselines"] if r["split"] == "main300" and r["version"] == version)
        values = [100 * row["metrics"][d] for d in dimensions]
        bars = ax.bar(x + (index - .5) * .35, values, .34, label=version.capitalize(), color=("#0072B2", "#E69F00")[index])
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=("Stress", "Feature exact", "Full joint", "Core joint"), ylim=(0, 109),
           ylabel="Agreement with stored labels (%)", title="Live verifiers versus stored automatic references")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.02))
    ax.grid(axis="y", alpha=.15)
    save(fig, "verifier_reference_diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "analysis").resolve()
    if output != root / "analysis" and root / "analysis" not in output.parents:
        raise ValueError("Outputs must stay in the study analysis/ directory")
    protocol = verify_freezes(root)
    if not args.allow_partial and (args.resamples != protocol["statistics"]["resamples"] or args.seed != protocol["statistics"]["bootstrap_seed"]):
        raise ValueError("Final analysis must use frozen resamples/seed")
    records, jobs, absent = collect_predictions(root, protocol, args.allow_partial)
    items = mean_replicates(records, protocol["replicates"])
    baselines = baseline_records(root)
    summary = build_summary(records, items, jobs, baselines, protocol, args.resamples, args.seed)
    summary.update(analysis_status="PARTIAL_DIAGNOSTIC_NOT_FINAL" if absent else "complete_planned_denominators",
                   missing_final_jobs=absent, generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   protocol_sha256=sha(root / "protocol_frozen.json"), analysis_code_sha256=sha(Path(__file__)),
                   historical_golden26_diagnostic=historical_diagnostic(root))
    if absent and output == root / "analysis":
        raise ValueError("Partial diagnostics need --output analysis/<distinct-subdirectory> to protect final result paths")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_prediction", records), ("per_item", items), ("jobs", jobs), ("baseline_items", baselines)):
        write_json(output / (name + ".json"), rows)
        write_csv(output / (name + ".csv"), rows)
    write_json(output / "summary.json", summary)
    latex_export(summary, output / "paper_results.tex")
    if not args.no_figures:
        create_figures(summary, output / "figures")
    print(json.dumps({"status": summary["analysis_status"], "prediction_records": len(records),
                      "missing_jobs": len(absent), "output": str(output)}))


if __name__ == "__main__":
    main()
