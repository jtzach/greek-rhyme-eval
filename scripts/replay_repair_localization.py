"""Post hoc, descriptive V/W localization; standard-library offline public replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = "extension_2026-10-10"
ORIGINAL = "study_2026-10-08"
OUTPUT = "posthoc_repair_localization.json"
MODELS = ("astra", "sol", "luna", "claude_sonnet55", "claude_haiku55", "gemini38_flash",
          "deepseek41_flash", "qwen38_flash", "mistral_large4", "llama4_maverick", "glm53_flash")
CHANGED = "structured_feedback_changed"
UNCHANGED = "structured_feedback_unchanged"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(row):
    return (row["predicted_type"], tuple(sorted(row["predicted_features"]))) if row["valid"] else None


def correctness(row, core=False):
    if not row["valid"]:
        return 0
    predicted, reference = set(row["predicted_features"]), set(row["reference_features"])
    if core:
        predicted -= {"COPY", "MOSAIC"}
        reference -= {"COPY", "MOSAIC"}
    return int(row["predicted_type"] == row["reference_type"] and predicted == reference)


def recompute(records, groups):
    """Replay exclusively from categorical prediction rows and opaque group IDs."""
    require(set(groups) == {CHANGED, UNCHANGED}, "Unexpected localization groups")
    changed, unchanged = set(groups[CHANGED]), set(groups[UNCHANGED])
    require(len(groups[CHANGED]) == len(changed) == 50 and len(groups[UNCHANGED]) == len(unchanged) == 250,
            "Changed/unchanged group denominators differ")
    require(not changed & unchanged, "Localization groups overlap")
    identifiers = changed | unchanged
    require(identifiers == {f"pair_{i:03d}" for i in range(1, 301)}, "Unexpected main IDs")
    index, references = {}, {}
    for row in records:
        if row["split"] != "main300" or row["condition"] not in ("V", "W"):
            continue
        key = row["model"], row["condition"], row["replicate"], row["id"]
        require(key not in index and row["model"] in MODELS and row["replicate"] in (1, 2)
                and row["id"] in identifiers, "Duplicate/unexpected prediction")
        require(row["valid"] in (0, 1), "Invalid validity indicator")
        reference = row["reference_type"], tuple(sorted(row["reference_features"]))
        require(references.setdefault(row["id"], reference) == reference, "References differ across records")
        require(correctness(row) == row["joint"] and correctness(row, True) == row["core_joint"],
                "Saved score differs from categorical labels")
        index[key] = row
    require(len(index) == 11 * 2 * 2 * 300, "Incomplete planned V/W records")
    results = []
    for model in MODELS:
        subsets = {}
        for group, ids in ((CHANGED, changed), (UNCHANGED, unchanged), ("all", identifiers)):
            total = {"items": len(ids), "paired_predictions": 2 * len(ids), "v_valid": 0, "w_valid": 0,
                "both_valid": 0, "both_invalid": 0, "validity_gains": 0, "validity_losses": 0,
                "both_valid_output_changes": 0, "validity_or_label_changes": 0,
                "v_joint_matches": 0, "w_joint_matches": 0, "joint_gains": 0, "joint_losses": 0,
                "both_valid_joint_gains": 0, "both_valid_joint_losses": 0,
                "v_core_matches": 0, "w_core_matches": 0}
            for item in sorted(ids):
                for rep in (1, 2):
                    v, w = index[model, "V", rep, item], index[model, "W", rep, item]
                    vv, wv = bool(v["valid"]), bool(w["valid"])
                    vj, wj = correctness(v), correctness(w)
                    both = vv and wv
                    total["v_valid"] += vv
                    total["w_valid"] += wv
                    total["both_valid"] += both
                    total["both_invalid"] += not vv and not wv
                    total["validity_gains"] += not vv and wv
                    total["validity_losses"] += vv and not wv
                    total["both_valid_output_changes"] += both and canonical(v) != canonical(w)
                    total["validity_or_label_changes"] += canonical(v) != canonical(w)
                    total["v_joint_matches"] += vj
                    total["w_joint_matches"] += wj
                    total["joint_gains"] += wj > vj
                    total["joint_losses"] += wj < vj
                    total["both_valid_joint_gains"] += both and wj > vj
                    total["both_valid_joint_losses"] += both and wj < vj
                    total["v_core_matches"] += correctness(v, True)
                    total["w_core_matches"] += correctness(w, True)
            total["net_joint_matches"] = total["w_joint_matches"] - total["v_joint_matches"]
            total["joint_delta_pp"] = 100 * total["net_joint_matches"] / total["paired_predictions"]
            total["contribution_to_overall_joint_delta_pp"] = 100 * total["net_joint_matches"] / 600
            total["core_delta_pp"] = 100 * (total["w_core_matches"] - total["v_core_matches"]) / total["paired_predictions"]
            subsets[group] = total
        results.append({"model": model, "model_requested": index[model, "V", 1, "pair_001"]["model_requested"],
                        "groups": subsets})
    return results


def build_local(root=ROOT):
    root = Path(root)
    protocol = read(root / EXT / "protocol_frozen.json")
    source_analysis = root / ORIGINAL / "design/verifier_analyses.json"
    require(sha(source_analysis) == protocol["original_hashes"]["design/verifier_analyses.json"],
            "Frozen verifier analyses changed")
    analyses = read(source_analysis)
    require(analyses == read(root / EXT / "design/verifier_analyses.json"), "Extension verifier analyses differ")
    # Group membership is determined from frozen verifier records BEFORE model
    # prediction files are opened. Selection of this analysis itself is post hoc.
    ids = {f"pair_{i:03d}" for i in range(1, 301)}
    changed = sorted(i for i in ids if analyses["original"][i]["raw_feedback"] != analyses["repaired"][i]["raw_feedback"])
    difference = root / ORIGINAL / "design/main_verifier_difference_audit.json"
    audit = read(difference)
    require(audit["n"] == 300 and changed == sorted(audit["raw_feedback_changes"]) and len(changed) == 50,
            "Frozen feedback-change audit does not agree")
    groups = {CHANGED: changed, UNCHANGED: sorted(ids - set(changed))}
    paths = [source_analysis, difference, root / EXT / "design/verifier_analyses.json"]
    records, summaries = [], []
    final_audit = read(root / EXT / "final_audit.json")
    require(final_audit.get("passed") is True and final_audit.get("failed_checks") == 0, "Final audit must pass")
    for study in (ORIGINAL, EXT):
        summary_path, prediction_path = root / study / "analysis/summary.json", root / study / "analysis/per_prediction.json"
        summary = read(summary_path)
        require(summary["analysis_status"] == "complete_planned_denominators" and summary["missing_final_jobs"] == []
                and summary["protocol_sha256"] == sha(root / study / "protocol_frozen.json"), "Incomplete or unbound final study")
        if study == EXT:
            require(final_audit["input_hashes"][EXT + "/analysis/per_prediction.json"] == sha(prediction_path), "Audited predictions changed")
        records += read(prediction_path)
        summaries.append(summary)
        paths += [summary_path, prediction_path, root / study / "protocol_frozen.json"]
    results = recompute(records, groups)
    for row in results:
        contrast = next(c for s in summaries for c in s["contrasts"] if c["model"] == row["model"] and c["contrast"] == "W-V")
        require(abs(row["groups"]["all"]["joint_delta_pp"] - 100 * contrast["joint_batch"]["estimate"]) < 1e-10,
                "Localization does not sum to the unchanged primary contrast")
    return {"analysis_type": "post_hoc_descriptive_repair_localization", "selection":
        "Requested after observing the completed W-V results, especially the Llama contrast; no confirmatory test or mechanism claim.",
        "group_definition": "Exact inequality of original versus repaired raw_feedback objects in the verifier analyses frozen before model inference; independently matches the saved 50-ID audit.",
        "groups": groups, "models": results,
        "semantics": {"unit": "Item-repetition comparison; two repeats per item, not independent items",
            "denominators": "All planned V/W records remain; changed group100, unchanged group500, overall600 per model",
            "output_changes": "Canonical type/features differ among two valid outputs; validity transitions are reported separately. Two invalid outputs are not a linguistic change.",
            "joint_changes": "Full automatic-reference exact agreement; both-valid gains/losses separate changes involving validity",
            "contribution": "Each subgroup net match count divided by600; subgroup contributions sum to the frozen overall W-V percentage-point estimate",
            "group_scope": "Unchanged means only the raw_feedback objects are equal. All300 full verifier objects differ because their version fields differ; other payload fields and mixed-batch context can also differ. This is not a group of identical full payloads or prompts. Public replay verifies counts conditional on published group IDs; it cannot regenerate membership without the omitted verifier objects.",
            "limits": "Post hoc localization only; fixed batches mix groups and may transmit feedback effects across items. Sampling variation, contextual interactions and model behavior are not causally isolated. No new confidence intervals, significance tests, relabeling or main-metric changes."},
        "source_sha256": {p.relative_to(root).as_posix(): sha(p) for p in paths},
        "analysis_code_sha256": sha(Path(__file__))}


def verify_public(root):
    root = Path(root)
    saved = read(root / "results" / EXT / OUTPUT)
    require(saved["analysis_type"] == "post_hoc_descriptive_repair_localization", "Wrong post hoc artifact")
    rows = read(root / "results/per_prediction.json") + read(root / "results" / EXT / "per_prediction.json")
    require(recompute(rows, saved["groups"]) == saved["models"], "Public localization replay differs")
    require(saved["analysis_code_sha256"] == sha(Path(__file__)), "Localization replay code differs")
    return {"status": "verified", "models": 11, "changed_items": 50, "unchanged_items": 250}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    require(not (args.public_root and args.write), "Public replay is read only")
    if args.public_root:
        print(json.dumps(verify_public(args.public_root)))
        return
    result = build_local()
    if args.write:
        target = ROOT / EXT / "analysis" / OUTPUT
        data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        require(not target.exists() or target.read_bytes() == data, "Existing post hoc artifact differs; never overwrite")
        target.write_bytes(data)
    for model in result["models"]:
        print(json.dumps({"model": model["model"], **{g: {k: v[k] for k in ("v_joint_matches", "w_joint_matches", "net_joint_matches", "both_valid_output_changes", "validity_gains", "validity_losses", "contribution_to_overall_joint_delta_pp")} for g, v in model["groups"].items()}}))


if __name__ == "__main__":
    main()
