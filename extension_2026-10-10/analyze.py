"""Offline extension scoring using the original frozen metric implementation.

No inference and no output repair. Outputs remain inside this extension's
analysis directory. Partial diagnostic output needs a distinct subdirectory.
The stored references are automatic labels, not independent human gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = load_module(ROOT / "prepare.py", "extension_preparation_analysis")
OLD = load_module(ROOT.parent / "study_2026-10-08/analysis.py", "frozen_original_analysis_extension")
read, sha = PREP.read, PREP.sha


def verify(root=ROOT):
    protocol = PREP.verify_freeze(root)
    original = root.parent / "study_2026-10-08"
    reference_path = original / "data/references/main300.json"
    expected = read(original / "data/freeze.json")["files_sha256"]["data/references/main300.json"]
    if sha(reference_path) != expected:
        raise ValueError("Original reference labels changed")
    implementation = read(original / "analysis/implementation_freeze.json")
    if sha(original / "analysis.py") != implementation["files_sha256"]["analysis.py"]:
        raise ValueError("Original frozen metric implementation changed")
    phase2 = root / "design/phase2_freeze.json"
    if (root / "manifests/phase2.json").exists() and not phase2.exists():
        raise ValueError("Unfrozen second-pass manifest")
    if phase2.exists():
        value = read(phase2)
        if value["protocol_sha256"] != sha(root / "protocol_frozen.json"):
            raise ValueError("Phase 2 used a different extension protocol")
        for relative, expected in value["hashes"].items():
            if sha(root / relative) != expected:
                raise ValueError("Frozen second-pass artifact changed: " + relative)
        for dependency, expected in value["dependencies"].items():
            folder = root / "runs/phase1" / dependency
            if sha(folder / "response.txt") != expected or read(folder / "meta.json").get("response_sha256") != expected:
                raise ValueError("Frozen shared S response changed: " + dependency)
    return protocol


def api_cost_record(folder, meta):
    record = OLD.cost_record(folder, meta)
    attempts = [read(p) for p in sorted(folder.glob("attempt_*/meta.json"))]
    if not attempts and meta is not None:
        attempts = [meta]
    # The API's cost field is recorded spending, not a subscription-cost proxy.
    values = []
    for attempt in attempts:
        usage = attempt.get("usage") or {}
        value = usage.get("cost", attempt.get("cost_usd"))
        if value is not None:
            values.append(float(value))
    record.update(reported_cost_usd=sum(values) if values else None,
                  attempts_reporting_cost=len(values), cost_is_complete=len(values) == len(attempts))
    return record


def check_run_config(root, phase, protocol, runner):
    path = root / "runs" / phase / "run_config.json"
    if not path.exists():
        return None
    config = read(path)
    for key, expected in (("runner_sha256", protocol["hashes"]["runner.py"]),
                          ("manifest_sha256", sha(root / "manifests" / (phase + ".json"))),
                          ("wrapper_sha256", PREP.digest(runner.WRAPPER))):
        if config.get(key) != expected:
            raise ValueError("Run configuration differs: " + phase + "/" + key)
    manifest = read(root / "manifests" / (phase + ".json"))
    if manifest["models"] != protocol["model_settings"]:
        raise ValueError("Requested model settings differ from protocol")
    if config.get("models") != protocol["model_settings"]:
        raise ValueError("Run model settings differ from frozen manifest")
    return config


def collect_predictions(root, protocol, allow_partial=False):
    runner = load_module(root / "runner.py", "extension_runner_analysis")
    references = {r["id"]: r for r in read(root.parent / "study_2026-10-08/data/references/main300.json")}
    configs = {phase: check_run_config(root, phase, protocol, runner) for phase in ("phase1", "phase2")}
    all_jobs = OLD.planned_jobs(root, protocol["models"])
    first_jobs = {j["id"]: j for j in all_jobs if j["phase"] == "phase1"}
    records, jobs, absent, label_only = [], [], [], []
    for job in all_jobs:
        folder = root / "runs" / job["phase"] / job["id"]
        meta = read(folder / "meta.json") if (folder / "meta.json").exists() else None
        predictions = {}
        status = "blocked_by_initial_failure" if job["blocked"] else "missing_final_metadata"
        if job["phase"] == "phase2" and (root / "manifests/phase2.json").exists():
            initial, _, initial_valid = PREP.committed_response(root, first_jobs[job["dependency"]], protocol, runner, configs["phase1"])
            if job["blocked"] != (not initial_valid):
                raise ValueError("Blocked revision contradicts its shared S output")
            if not job["blocked"] and job.get("dependency_response_sha256") != initial["response_sha256"]:
                raise ValueError("Revision shared S response changed")
            if job["blocked"] and meta is not None:
                raise ValueError("Blocked revision unexpectedly has inference output")
        if meta is not None:
            response_path = folder / "response.txt"
            response = response_path.read_bytes().decode("utf-8") if response_path.exists() else ""
            if PREP.digest(response) != meta.get("response_sha256"):
                raise ValueError("Response hash differs: " + job["id"])
            config = configs[job["phase"]]
            if config is None:
                raise ValueError("Missing run configuration: " + job["phase"])
            if meta.get("id") != job["id"] or meta.get("model_requested") != protocol["models"][job["model_key"]]:
                raise ValueError("Saved job/model identity differs: " + job["id"])
            prompt_path = (root / "manifests" / job["prompt_file"]).resolve()
            if sha(prompt_path) != config["prompt_hashes"].get(job["id"]):
                raise ValueError("Prompt hash mismatch: " + job["id"])
            prompt = runner.WRAPPER + prompt_path.read_text(encoding="utf-8")
            if PREP.digest(prompt) != meta.get("prompt_sha256") or (folder / "input.txt").read_text(encoding="utf-8") != prompt:
                raise ValueError("Saved sent input differs: " + job["id"])
            payload, _, _ = runner.build_request(job, protocol["model_settings"][job["model"]], prompt)
            request_hash = hashlib.sha256(runner.encode(payload)).hexdigest()
            if meta.get("request_sha256") != request_hash or config["request_hashes"].get(job["id"]) != request_hash:
                raise ValueError("Saved request differs from frozen model settings")
            if (folder / "request.json").exists() and sha(folder / "request.json") != request_hash:
                raise ValueError("Saved serialized request changed")
            if meta.get("raw_response_sha256") is not None and sha(folder / "raw_response.json") != meta["raw_response_sha256"]:
                raise ValueError("Saved raw API response changed")
            format_ok, _ = runner.validate_answer(response, job["item_ids"])
            inference_ok = meta.get("returncode") == 0 and meta.get("completed") is True and meta.get("tool_event_count") == 0
            valid = bool(meta.get("valid") and inference_ok and format_ok)
            if bool(meta.get("valid")) != valid:
                raise ValueError("Metadata validity contradicts independent validation")
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
        costs = api_cost_record(folder, meta)
        jobs.append({"job_id": job["id"], "phase": job["phase"], "split": "main300", "model": job["model_key"],
                     "model_requested": protocol["models"][job["model_key"]], "condition": job["condition"],
                     "replicate": job["replicate"], "batch": job["batch"], "items": len(job["item_ids"]),
                     "status": status, "dependency": job.get("dependency"), **costs})
        label_only.append({"job_id": job["id"], "model": job["model_key"], "model_requested": protocol["models"][job["model_key"]],
                           "condition": job["condition"], "replicate": job["replicate"], "batch": job["batch"],
                           "status": status, "expected_item_ids": job["item_ids"],
                           "predictions": list(predictions.values()), "response_sha256": meta.get("response_sha256") if meta else None})
        for item_id in job["item_ids"]:
            source = references[item_id]
            reference, prediction = source["reference"], predictions.get(item_id)
            record = {"id": item_id, "split": "main300", "model": job["model_key"], "condition": job["condition"],
                      "model_requested": protocol["models"][job["model_key"]], "replicate": job["replicate"], "batch": job["batch"],
                      "group": source["provenance"].get("grouping_key") or "UNRESOLVED_SOURCE_GROUP",
                      "job_id": job["id"], "status": status, "reference_type": reference["position"],
                      "reference_features": sorted(reference["features"]), "predicted_type": prediction["type"] if prediction else None,
                      "predicted_features": sorted(prediction["features"]) if prediction else [],
                      "response_sha256": meta.get("response_sha256") if meta else None}
            record.update(OLD.score_prediction(reference, prediction, valid=status == "valid"))
            records.append(record)
    expected = protocol["counts"]["main_prediction_records"]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} planned records; found {len(records)}")
    if not allow_partial and (absent or not (root / "manifests/phase2.json").exists()):
        raise RuntimeError(f"{len(absent)} uncommitted batches; complete phases or request --allow-partial")
    return records, jobs, absent, label_only


def summarize_cost(jobs):
    result = OLD.summarize_cost(jobs)
    values = [j["reported_cost_usd"] for j in jobs if j["reported_cost_usd"] is not None]
    result.update(reported_cost_usd=sum(values) if values else None,
                  attempts_reporting_cost=sum(j["attempts_reporting_cost"] for j in jobs),
                  jobs_with_incomplete_cost=sum(not j["cost_is_complete"] for j in jobs))
    return result


def build_summary(records, items, jobs, protocol, resamples=10000, seed=20261008):
    summary = {"reference_status": "automatic stored corpus labels; agreement, not human-validated accuracy",
               "counts": {"prediction_records": len(records), "item_condition_records": len(items)},
               "statistics": {"resamples": resamples, "seed": seed,
                              "repetition_policy": "average two outcomes within item before resampling the same 30 frozen batches",
                              "intervals": "Descriptive paired 95% percentile intervals without multiplicity adjustment; no significance declarations"},
               "metric_notes": {"feature_components": "Feature sets are scored irrespective of type; joint requires stress and feature-set agreement. Invalid outputs contribute all reference features as FN.",
                                "core_joint": "Remove COPY and MOSAIC from prediction and reference before exact joint comparison.",
                                "empty_sets": "Valid empty/empty item F1=1; invalid F1=0; undefined aggregate micro F1=null.",
                                "stability": "Both repeats must be valid with identical type/set. Different frozen E demonstrations confound decoding with example variability.",
                                "cost": "Physical calls counted once in grand total. Standalone R/V/W adds the reused S call. Cost is reported API USD, with missingness retained. Latency sums measured stages, excluding queue time."},
               "scope": protocol["scope"], "cells": [], "contrasts": [], "cost_total": summarize_cost(jobs)}
    for model, model_id in protocol["models"].items():
        for condition in PREP.CONDITIONS:
            predictions = [r for r in records if r["model"] == model and r["condition"] == condition]
            subset = [r for r in items if r["model"] == model and r["condition"] == condition]
            own_jobs = [j for j in jobs if j["model"] == model and j["condition"] == condition]
            standalone = own_jobs + ([j for j in jobs if j["model"] == model and j["condition"] == "S"] if condition in ("R", "V", "W") else [])
            if len(subset) != 300 or len(predictions) != 600 or len({r["batch"] for r in subset}) != 30:
                raise ValueError("Incomplete planned cell denominator")
            both = sum(r["both_replicates_valid"] for r in subset)
            stable = sum(r["stable_valid_label_set"] for r in subset)
            summary["cells"].append({"split": "main300", "model": model, "model_requested": model_id, "condition": condition,
                "n_items": len(subset), "n_predictions": len(predictions), "metrics": OLD.summarize_metrics(predictions),
                "by_position": {p: OLD.summarize_metrics([r for r in predictions if r["reference_type"] == p]) for p in ("M", "F2", "F3")},
                "intervals": {m: OLD.cluster_bootstrap([r[m] for r in subset], [r["batch"] for r in subset], resamples, seed) for m in OLD.METRICS},
                "author_group_joint_sensitivity": OLD.cluster_bootstrap([r["joint"] for r in subset], [r["group"] for r in subset], resamples, seed),
                "stability": {"both_valid": both, "stable_valid": stable, "denominator": 300,
                              "unconditional_rate": stable / 300, "conditional_on_both_valid": OLD.ratio(stable, both)},
                "cost_incremental": summarize_cost(own_jobs), "cost_standalone_condition": summarize_cost(standalone),
                "standalone_pipeline_latency": OLD.pipeline_latency(own_jobs, jobs)})
        for a, b in OLD.CONTRASTS:
            lhs = [r for r in items if r["model"] == model and r["condition"] == a]
            rhs = [r for r in items if r["model"] == model and r["condition"] == b]
            summary["contrasts"].append({"model": model, "contrast": a + "-" + b,
                "joint_batch": OLD.paired_contrast(lhs, rhs, "batch", resamples, seed),
                "joint_author_group_sensitivity": OLD.paired_contrast(lhs, rhs, "group", resamples, seed),
                "core_joint_batch": OLD.paired_contrast(lhs, rhs, "batch", resamples, seed, "core_joint")})
    return summary


def compare_original(root, summary, items, resamples, seed):
    """Read archived results only; never invoke the original analysis main()."""
    original = root.parent / "study_2026-10-08"
    source = original / "analysis/summary.json"
    item_source = original / "analysis/per_item.json"
    archived = read(source)
    old_items = [r for r in read(item_source) if r["split"] == "main300"]
    old_cells = [c for c in archived["cells"] if c["split"] == "main300"]
    metadata = {"interface_caveat": "Direct OpenRouter API and Codex CLI differ; differences are descriptive and cannot isolate a model-family effect.",
                "source_summary_sha256": sha(source), "source_items_sha256": sha(item_source),
                "original_analysis_status": archived["analysis_status"], "cells": [], "matched_condition_contrasts": []}
    for origin, cells in (("original_codex", old_cells), ("extension_openrouter", summary["cells"])):
        metadata["cells"] += [{"study": origin, **c} for c in cells]
    for model in dict.fromkeys(r["model"] for r in items):
        for old_model in dict.fromkeys(r["model"] for r in old_items):
            for condition in PREP.CONDITIONS:
                lhs = [r for r in items if r["model"] == model and r["condition"] == condition]
                rhs = [r for r in old_items if r["model"] == old_model and r["condition"] == condition]
                for newer, older in zip(sorted(lhs, key=lambda r: r["id"]), sorted(rhs, key=lambda r: r["id"])):
                    for key in ("id", "batch", "reference_type", "reference_features"):
                        if newer[key] != older[key]:
                            raise ValueError("Archived comparison reference/batch differs")
                metadata["matched_condition_contrasts"].append({"new_model": model, "original_model": old_model,
                    "condition": condition, "contrast": "new minus original",
                    "joint_batch": OLD.paired_contrast(lhs, rhs, "batch", resamples, seed),
                    "core_joint_batch": OLD.paired_contrast(lhs, rhs, "batch", resamples, seed, "core_joint")})
    return metadata


def report(summary, output):
    def pct(value):
        return "—" if value is None else f"{100 * value:.1f}"
    lines = ["# Expanded Greek rhyme evaluation", "", f"Status: **{summary['analysis_status']}**.", "",
             "Scores measure agreement with stored automatic corpus labels, not human-validated accuracy. All 300 pairs and two repetitions remain in every condition denominator; missing, malformed, tool-using and dependent failures count as errors.", "",
             "The original models were tested through Codex CLI; the extension uses OpenRouter's direct API. Cross-model differences also reflect interface, provider serving and reasoning-setting differences.", "",
             "| Model | Condition | Joint % | Stress % | Feature exact % | Micro F1 % | Item F1 % | Core joint % | Valid % |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in summary["cells"]:
        values = [pct(c["metrics"][m]) for m in ("joint", "stress", "feature_exact", "feature_micro_f1", "feature_item_f1", "core_joint", "valid")]
        lines.append("| " + " | ".join([c["model_requested"], c["condition"]] + values) + " |")
    lines += ["", "| Model | Contrast | Joint difference (pp) | Descriptive 95% batch interval (pp) |", "|---|---|---:|---:|"]
    for row in summary["contrasts"]:
        v = row["joint_batch"]
        lines.append(f"| {row['model']} | {row['contrast']} | {100*v['estimate']:+.1f} | [{100*v['lower']:+.1f}, {100*v['upper']:+.1f}] |")
    lines += ["", "Two repetitions are averaged within each item before paired resampling of the same 30 fixed ten-pair batches. These descriptive intervals are not multiplicity-adjusted significance tests. Per-position scores, feature F1, core agreement, repeat stability, source-group sensitivity, latency and recorded API costs are in summary.json.", "",
              "COPY and MOSAIC are excluded from both feature sets only in the prespecified core sensitivity metric. Verifier feedback is related to the system that produced the references, so agreement gains are not independent accuracy gains. The sample contains positive automatic references and balanced stress classes; it does not measure general rhyme detection or corpus prevalence. Author/source groups are not known poem groups.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--skip-original-comparison", action="store_true")
    args = parser.parse_args()
    root = ROOT.resolve()
    output = (args.output or root / "analysis").resolve()
    if output != root / "analysis" and root / "analysis" not in output.parents:
        raise ValueError("Outputs must stay inside extension analysis/")
    protocol = verify(root)
    if args.resamples <= 0:
        raise ValueError("Positive resample count required")
    if not args.allow_partial and (args.resamples != protocol["statistics"]["resamples"] or args.seed != protocol["statistics"]["bootstrap_seed"]):
        raise ValueError("Final analysis must use frozen seed and resample count")
    if args.allow_partial and output == root / "analysis":
        raise ValueError("Partial diagnostics require --output extension_2026-10-10/analysis/<distinct-subdirectory>")
    records, jobs, absent, label_only = collect_predictions(root, protocol, args.allow_partial)
    items = OLD.mean_replicates(records, 2)
    summary = build_summary(records, items, jobs, protocol, args.resamples, args.seed)
    summary.update(analysis_status="PARTIAL_DIAGNOSTIC_NOT_FINAL" if args.allow_partial or absent else "complete_planned_denominators",
                   missing_final_jobs=absent, generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   protocol_sha256=sha(root / "protocol_frozen.json"), analysis_code_sha256=sha(Path(__file__)))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_prediction", records), ("per_item", items), ("jobs", jobs), ("predictions_label_only", label_only)):
        OLD.write_json(output / (name + ".json"), rows)
        OLD.write_csv(output / (name + ".csv"), rows)
    if not args.skip_original_comparison:
        comparison = compare_original(root, summary, items, args.resamples, args.seed)
        OLD.write_json(output / "comparison_original.json", comparison)
    OLD.write_json(output / "summary.json", summary)
    report(summary, output)
    print(json.dumps({"status": summary["analysis_status"], "planned_records": len(records), "missing_jobs": len(absent), "output": str(output)}))


if __name__ == "__main__":
    main()
