"""Offline matched comparison of OpenRouter and archived Codex configurations.

No API calls. No archived result is changed. All paired intervals use the
original frozen metric implementation and the same 300 items / 30 batches.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "study_2026-10-08"
PAIRS = {"astra_api": "astra", "sol_api": "sol", "luna_api": "luna"}
CONDITIONS = ("S", "C", "E", "R", "V", "W")
METRICS = ("joint", "stress", "feature_exact", "core_joint", "valid")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_items(items, model, condition):
    selected = [r for r in items if r["split"] == "main300" and
                r["model"] == model and r["condition"] == condition]
    if len(selected) != 300 or len({r["id"] for r in selected}) != 300:
        raise ValueError(f"Expected 300 unique matched items: {model}/{condition}")
    if Counter(r["batch"] for r in selected) != Counter({b: 10 for b in range(1, 31)}):
        raise ValueError("Expected the same 30 ten-pair inference batches")
    if any(r["replicates"] != 2 for r in selected):
        raise ValueError("Expected two repetitions averaged within every item")
    return selected


def cells_by_key(summary, expected_models):
    if summary.get("analysis_status") != "complete_planned_denominators":
        raise ValueError("Matched comparison requires final, complete main analyses")
    if summary.get("missing_final_jobs"):
        raise ValueError("Uncommitted jobs remain in source analysis")
    cells = [c for c in summary["cells"] if c["split"] == "main300"]
    result = {(c["model"], c["condition"]): c for c in cells}
    expected = {(m, c) for m in expected_models for c in CONDITIONS}
    if len(result) != len(cells) or set(result) != expected:
        raise ValueError("Source main-study cells do not match the declared configurations")
    if any(c["n_items"] != 300 or c["n_predictions"] != 600 for c in cells):
        raise ValueError("Each condition must retain 300 items and 600 planned predictions")
    return result


def build_comparison(api_summary, api_items, api_jobs, codex_summary,
                     codex_items, codex_jobs, metric_impl, resamples, seed):
    api_cells = cells_by_key(api_summary, PAIRS)
    codex_cells = cells_by_key(codex_summary, PAIRS.values())
    comparisons, reliability = [], []
    for api_model, codex_model in PAIRS.items():
        for condition in CONDITIONS:
            api_cell = api_cells[(api_model, condition)]
            codex_cell = codex_cells[(codex_model, condition)]
            if api_cell["model_requested"] != "openai/" + codex_cell["model_requested"]:
                raise ValueError("Matched model names differ beyond the OpenRouter namespace")
            api = select_items(api_items, api_model, condition)
            codex = select_items(codex_items, codex_model, condition)
            api_ids, codex_ids = {r["id"]: r for r in api}, {r["id"]: r for r in codex}
            if api_ids.keys() != codex_ids.keys():
                raise ValueError("Cross-interface item IDs differ")
            for item_id in api_ids:
                for key in ("batch", "group", "reference_type", "reference_features"):
                    if api_ids[item_id][key] != codex_ids[item_id][key]:
                        raise ValueError(f"Cross-interface item contract differs: {item_id}/{key}")
            contrasts = {metric: metric_impl.paired_contrast(
                api, codex, "batch", resamples, seed, metric) for metric in METRICS}
            for metric in METRICS:
                expected = api_cell["metrics"][metric] - codex_cell["metrics"][metric]
                if abs(contrasts[metric]["estimate"] - expected) > 1e-12:
                    raise ValueError("Source summary and paired item estimate disagree")
            comparisons.append({
                "alias": codex_model, "api_model_key": api_model,
                "codex_model_key": codex_model, "condition": condition,
                "api_requested_id": api_cell["model_requested"],
                "codex_requested_id": codex_cell["model_requested"],
                "n_items": 300, "n_predictions_per_interface": 600,
                "api_metrics": api_cell["metrics"], "codex_metrics": codex_cell["metrics"],
                "difference_direction": "OpenRouter API minus archived Codex",
                "paired_batch_differences": contrasts,
                "joint_author_group_sensitivity": metric_impl.paired_contrast(
                    api, codex, "group", resamples, seed, "joint"),
            })
        for interface, model, jobs in (("OpenRouter API", api_model, api_jobs),
                                       ("Codex CLI", codex_model, codex_jobs)):
            subset = [j for j in jobs if j["split"] == "main300" and j["model"] == model]
            if len(subset) != 360 or len({j["job_id"] for j in subset}) != 360:
                raise ValueError("Expected 360 planned unique batch jobs per configuration")
            if Counter(j["condition"] for j in subset) != Counter({c: 60 for c in CONDITIONS}):
                raise ValueError("Reliability cells must retain 60 planned jobs per condition")
            reliability.append({
                "alias": codex_model, "model_key": model, "interface": interface,
                "planned_batch_jobs": len(subset),
                "status_counts": dict(sorted(Counter(j["status"] for j in subset).items())),
                "recorded_attempt_entries": sum(j["attempts"] for j in subset),
                "condition_status_counts": {
                    c: dict(sorted(Counter(j["status"] for j in subset if j["condition"] == c).items()))
                    for c in CONDITIONS},
            })
    return {"status": "complete_matched_configuration_comparison", "cells": comparisons,
            "reliability": reliability,
            "statistics": {"resamples": resamples, "seed": seed,
                "unit": "Two repetitions averaged within each item; paired resampling of 30 ten-pair batches",
                "intervals": "Descriptive 95% percentile intervals; no multiplicity-adjusted inference",
                "direction": "OpenRouter API minus archived Codex"},
            "scope": [
                "Three additional serving configurations of the same three named model aliases, not three additional unique models.",
                "Requested alias correspondence does not independently attest identical underlying weights or immutable snapshots.",
                "Interface, run date, hidden system instructions, reasoning/output limits, provider serving and other configuration differences can contribute; this is not a randomized causal interface ablation.",
                "S/C/E reuse frozen input prompt bytes. R/V/W revise each configuration's own S answer, so the revision inputs can differ between configurations.",
                "Agreement with automatic stored corpus labels, not human-validated accuracy. Invalid outputs and blocked dependencies remain errors in the planned denominators.",
                "Pairing is by item/batch after averaging repetitions; repetition numbers do not establish matched random decoding draws.",
                "Recorded attempt entries may include interrupted reservations without verified HTTP dispatch. Codex subscription dollar costs are unavailable.",
            ]}


def table_rows(result):
    rows = []
    for cell in result["cells"]:
        row = {k: cell[k] for k in ("alias", "condition", "api_requested_id", "codex_requested_id")}
        for metric in METRICS:
            row[f"codex_{metric}"] = cell["codex_metrics"][metric]
            row[f"api_{metric}"] = cell["api_metrics"][metric]
            for field in ("estimate", "lower", "upper"):
                row[f"api_minus_codex_{metric}_{field}"] = cell["paired_batch_differences"][metric][field]
        rows.append(row)
    return rows


def render_report(result):
    lines = ["# Matched OpenRouter API and Codex comparison", "",
             "These are three additional serving configurations of the same three named aliases. All 300 items, two repetitions and six conditions remain in each configuration's planned denominators.", "",
             "Differences below are API minus Codex. They describe configuration differences and do not isolate a causal effect of the interface or establish identical model snapshots.", "",
             "| Alias | Condition | Codex joint % | API joint % | Difference (pp) | Descriptive 95% batch interval (pp) | Codex valid % | API valid % |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in result["cells"]:
        v = c["paired_batch_differences"]["joint"]
        lines.append(f"| {c['alias']} | {c['condition']} | {100*c['codex_metrics']['joint']:.1f} | {100*c['api_metrics']['joint']:.1f} | {100*v['estimate']:+.1f} | [{100*v['lower']:+.1f}, {100*v['upper']:+.1f}] | {100*c['codex_metrics']['valid']:.1f} | {100*c['api_metrics']['valid']:.1f} |")
    lines += ["", "## Batch reliability", "",
              "| Alias | Interface | Planned jobs | Valid jobs | Invalid completed jobs | Transport/interrupted | Blocked revisions | Recorded attempt entries |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["reliability"]:
        statuses = r["status_counts"]
        valid = statuses.get("valid", 0)
        transport = statuses.get("transport_or_interrupted_failure", 0)
        blocked = statuses.get("blocked_by_initial_failure", 0)
        invalid = r["planned_batch_jobs"] - valid - transport - blocked
        lines.append(f"| {r['alias']} | {r['interface']} | {r['planned_batch_jobs']} | {valid} | {invalid} | {transport} | {blocked} | {r['recorded_attempt_entries']} |")
    lines += ["", "## Interpretation", ""]
    lines += ["- " + note for note in result["scope"]]
    lines += ["", "Stress, feature exact, core joint and validity differences, source digests, and per-condition reliability are retained in the accompanying JSON/CSV. No previously archived estimate is replaced.", ""]
    return "\n".join(lines)


def main():
    analysis = load_module(ROOT / "analyze.py", "matched_api_analysis")
    protocol = analysis.verify(ROOT)
    if set(protocol["models"]) != set(PAIRS):
        raise ValueError("This comparison requires exactly the three frozen API configurations")
    implementation = read(ORIGINAL / "analysis/implementation_freeze.json")
    if sha(ORIGINAL / "analysis.py") != implementation["files_sha256"]["analysis.py"]:
        raise ValueError("Archived metric implementation changed")
    sources = {}
    loaded = {}
    for origin, folder in (("openrouter_api", ROOT), ("codex", ORIGINAL)):
        for filename in ("summary.json", "per_item.json", "jobs.json"):
            path = folder / "analysis" / filename
            loaded[(origin, filename)] = read(path)
            sources[f"{origin}/analysis/{filename}"] = sha(path)
        sources[f"{origin}/protocol_frozen.json"] = sha(folder / "protocol_frozen.json")
    api_summary = loaded[("openrouter_api", "summary.json")]
    if api_summary.get("protocol_sha256") != sha(ROOT / "protocol_frozen.json"):
        raise ValueError("API analysis belongs to a different protocol")
    if api_summary.get("analysis_code_sha256") != sha(ROOT / "analyze.py"):
        raise ValueError("API analysis implementation differs")
    result = build_comparison(api_summary, loaded[("openrouter_api", "per_item.json")],
        loaded[("openrouter_api", "jobs.json")], loaded[("codex", "summary.json")],
        loaded[("codex", "per_item.json")], loaded[("codex", "jobs.json")], analysis.OLD,
        protocol["statistics"]["resamples"], protocol["statistics"]["bootstrap_seed"])
    result.update(source_sha256=sources, comparison_code_sha256=sha(Path(__file__)),
                  python_version=platform.python_version(), generated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    output = ROOT / "analysis"
    analysis.OLD.write_json(output / "comparison_codex_matched.json", result)
    analysis.OLD.write_csv(output / "comparison_codex_matched.csv", table_rows(result))
    (output / "comparison_codex_matched.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "matched_cells": len(result["cells"])}))


if __name__ == "__main__":
    main()
