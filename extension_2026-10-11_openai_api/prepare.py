"""Outcome-independent extension preparation. Never opens stored reference labels.

Run ``python -B extension_2026-10-10/prepare.py freeze`` after the synthetic
access checks and panel selection, and ``... phase2`` after phase 1 completes.
Only files below this extension directory are written. Existing artifacts are
checked rather than overwritten. All original-study files remain read-only.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "study_2026-10-08"
CONDITIONS = ("S", "C", "E", "R", "V", "W")
SEED = 20261008


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_bytes(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refuse artifact mutation: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def save(path, value):
    save_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel_models(panel):
    """Return stable analysis keys and the runner's exact-ID settings mapping."""
    entries = panel.get("models")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("panel.models must be a nonempty mapping of safe keys")
    identities, settings = {}, {}
    for key, entry in entries.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Unsafe model key: " + key)
        model = entry.get("id", entry.get("model"))
        if not isinstance(model, str) or "/" not in model or model in settings:
            raise ValueError("Each panel entry needs a unique exact OpenRouter model ID")
        config = dict(entry["settings"])
        if "pricing" in entry:
            config["pricing"] = entry["pricing"]
        for field in ("max_tokens", "reasoning", "response_format", "pricing", "provider"):
            if field not in config:
                raise ValueError(f"Missing frozen setting {key}/{field}")
        if config["reasoning"] is None and config.get("reasoning_policy") != "unsupported":
            raise ValueError("Null reasoning requires explicit unsupported policy")
        for field in ("prompt", "completion"):
            if float(config["pricing"][field]) < 0:
                raise ValueError("Prices must be nonnegative USD/token")
        if not isinstance(config["max_tokens"], int) or config["max_tokens"] <= 0:
            raise ValueError("max_tokens must be a positive integer")
        provider = config["provider"]
        if not provider.get("only") or provider.get("allow_fallbacks") is not False or provider.get("require_parameters") is not True:
            raise ValueError("Freeze explicit provider routing without fallbacks")
        identities[key] = model
        settings[model] = config
    return identities, settings


def original_contract(original=ORIGINAL):
    """Check input-only original freezes; reference files are deliberately unopened."""
    protocol = read(original / "protocol_frozen.json")
    for relative, expected in protocol["hashes"].items():
        if sha(original / relative) != expected:
            raise ValueError("Original protocol artifact changed: " + relative)
    dataset = read(original / "data/freeze.json")
    for relative in ("data/inputs/main300.json", "data/examples/context_bank60.json"):
        if sha(original / relative) != dataset["files_sha256"][relative]:
            raise ValueError("Original inference input changed: " + relative)
    main = read(original / "data/inputs/main300.json")
    if len(main) != 300 or len({x["id"] for x in main}) != 300:
        raise ValueError("Expected exact 300 unique frozen main pairs")
    return protocol, main


def prompt_language(original=ORIGINAL):
    """Extract BASE/render_inputs without importing verifier or executing preparation."""
    parsed = ast.parse((original / "prepare_experiments.py").read_text(encoding="utf-8"))
    selected = []
    for node in parsed.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BASE" for t in node.targets):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "render_inputs":
            selected.append(node)
    namespace = {"json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<frozen prompt language>", "exec"), namespace)
    return namespace["BASE"], namespace["render_inputs"]


def revision_prompt(planned, response, symbolic, original=ORIGINAL):
    base, render_inputs = prompt_language(original)
    text = base + render_inputs(planned["items"]) + "\nPREVIOUS MODEL ANSWER (may contain mistakes):\n" + response + "\n"
    if planned["condition"] == "R":
        text += "Reconsider your previous classification carefully using the definitions and the verse text. Correct any errors; retain answers you judge correct. Return the complete revised predictions only.\n"
    else:
        version = "original" if planned["condition"] == "V" else "repaired"
        feedback = [dict(id=x["id"], **symbolic[version][x["id"]]) for x in planned["items"]]
        text += "\nSYMBOLIC PHONOLOGICAL VERIFIER FEEDBACK (automatic, may be fallible):\n" + json.dumps(feedback, ensure_ascii=False) + "\nReconsider your previous classification using this feedback and the verse text. Correct any errors; retain answers you judge correct. Return the complete revised predictions only.\n"
    return text


def verify_freeze(root=ROOT):
    protocol = read(root / "protocol_frozen.json")
    for relative, expected in protocol["hashes"].items():
        if sha(root / relative) != expected:
            raise ValueError("Frozen extension artifact changed: " + relative)
    for relative, expected in protocol["original_hashes"].items():
        if sha(root.parent / "study_2026-10-08" / relative) != expected:
            raise ValueError("Frozen original artifact changed: " + relative)
    return protocol


def freeze(root=ROOT, original=ORIGINAL):
    if (root / "protocol_frozen.json").exists():
        protocol = verify_freeze(root)
        print(json.dumps({"status": "already_frozen_verified", **protocol["counts"]}))
        return protocol
    if any((root / "runs").glob("phase*/**/meta.json")):
        raise ValueError("Real-data output predates protocol freeze")
    original_protocol, main = original_contract(original)
    panel = read(root / "panel.json")
    identities, settings = panel_models(panel)
    runner = load_module(root / "runner.py", "extension_runner_freeze_validation")
    original_runner = load_module(original / "runner.py", "original_runner_freeze_validation")
    if runner.WRAPPER != original_runner.WRAPPER or runner.SCHEMA != original_runner.SCHEMA:
        raise ValueError("Runner wrapper/schema differs from the frozen original contract")
    for model, config in settings.items():
        runner.build_request({"id": "synthetic_config_validation", "model": model}, config,
                             runner.WRAPPER + "Synthetic configuration validation only.")
    original_jobs = read(original / "manifests/phase1.json")["jobs"]
    archived_prompt_hashes = read(original / "runs/phase1/run_config.json")["prompt_hashes"]
    # Pick the same outcome-independent template model throughout; prompts are
    # model-independent. Preserve replicate/batch/condition order exactly.
    selected = [j for j in original_jobs if j["model_key"] == "astra" and j["split"] == "main300"]
    if len(selected) != 180:
        raise ValueError("Expected 180 frozen S/C/E template jobs")
    input_by_id = {x["id"]: x for x in main}
    original_revisions = [p for p in read(original / "design/revision_plan.json") if p["model_key"] == "astra"]
    original_sources = set(original_protocol["hashes"])
    original_sources.update(("protocol_frozen.json", "data/inputs/main300.json", "analysis.py", "analysis/implementation_freeze.json",
                             "runs/phase1/run_config.json"))
    jobs, revision_plan, source_prompts = [], [], {}
    for job in selected:
        original_path = (original / "manifests" / job["prompt_file"]).resolve()
        if sha(original_path) != archived_prompt_hashes[job["id"]]:
            raise ValueError("Original prompt differs from archived inference configuration: " + job["id"])
        relative = original_path.relative_to(original).as_posix()
        original_sources.add(relative)
        expected_items = main[(job["batch"] - 1) * 10:job["batch"] * 10]
        if job["item_ids"] != [x["id"] for x in expected_items]:
            raise ValueError("Original batches differ from main300 ordering")
        for key, model in identities.items():
            jid = f"{key}_{job['condition']}_r{job['replicate']}_b{job['batch']:02d}"
            destination = root / "prompts" / (jid + ".txt")
            save_bytes(destination, original_path.read_bytes())
            jobs.append({**{k: v for k, v in job.items() if k not in ("id", "model", "model_key", "prompt_file")},
                         "id": jid, "model": model, "model_key": key, "prompt_file": "../prompts/" + destination.name})
            source_prompts[jid] = {"source": relative, "sha256": sha(original_path)}
    for planned in original_revisions:
        for key in identities:
            jid = f"{key}_{planned['condition']}_r{planned['replicate']}_b{planned['batch']:02d}"
            revision_plan.append({**planned, "id": jid, "model_key": key,
                                  "dependency": f"{key}_S_r{planned['replicate']}_b{planned['batch']:02d}"})
    if len(revision_plan) != 180 * len(identities):
        raise ValueError("Incomplete frozen revision plan")
    for p in revision_plan:
        if p["items"] != [input_by_id[x["id"]] for x in p["items"]]:
            raise ValueError("Revision inputs differ from frozen main300")
    save(root / "manifests/phase1.json", {"models": settings, "jobs": jobs})
    save(root / "design/revision_plan.json", revision_plan)
    save(root / "design/prompt_sources.json", source_prompts)
    for relative in ("design/example_draws.json", "design/verifier_analyses.json", "data/inputs/main300.json"):
        save_bytes(root / relative, (original / relative).read_bytes())
    required = ["prepare.py", "analyze.py", "runner.py", "panel.json", "manifests/phase1.json",
                "design/revision_plan.json", "design/prompt_sources.json", "design/example_draws.json",
                "design/verifier_analyses.json", "data/inputs/main300.json"]
    required += [p.relative_to(root).as_posix() for p in root.glob("catalog_*.json")]
    required += [name for name in ("panel_candidates.json", "preflight.py", "preflight_summary.json", "test_design_analysis.py", "run_pipeline.py", "compare_codex.py", "collect_catalog.py", "tests/test_runner.py")
                 if (root / name).is_file()]
    for directory in ("sources", "preflight"):
        required += [p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix not in (".pyc",)]
    required += [p.relative_to(root).as_posix() for p in (root / "prompts").glob("*.txt")]
    protocol = {
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "Frozen before real-data extension inference; synthetic preflight may precede this freeze",
        "models": identities, "model_settings": settings, "replicates": 2, "batch_size": 10, "main_pairs": 300,
        "conditions": original_protocol["conditions"],
        "counts": {"initial_calls": len(jobs), "revision_calls": len(revision_plan),
                   "total_planned_calls": len(jobs) + len(revision_plan),
                   "main_prediction_records": 3600 * len(identities), "legacy_prediction_records": 0},
        "statistics": original_protocol["statistics"], "primary_metric": original_protocol["primary_metric"],
        "secondary_metrics": original_protocol["secondary_metrics"], "denominators": original_protocol["denominators"],
        "empty_features": original_protocol["empty_features"],
        "revision_failure_policy": original_protocol["revision_failure_policy"],
        "replication": "Two fresh independent requests per batch; same frozen batches and examples as original; no score-adaptive changes",
        "ordering": "Original astra template order across replicate, batch, condition; panel declaration order inside each template slot",
        "transport": "OpenRouter direct chat-completions API; exact model/provider/settings frozen per model",
        "scope": [s for s in original_protocol["scope"] if "three models" not in s and "no dollar cost" not in s] + [
            "Original OpenAI-family results used Codex CLI; extension uses a direct API. Interface, serving and reasoning differences confound cross-model comparisons.",
            "Exact request model IDs do not independently attest immutable model weights; provider/model metadata are retained.",
            "Synthetic access checks may determine access/settings before this freeze; panel/labels/settings cannot change in response to main scores."],
        "hashes": {relative: sha(root / relative) for relative in sorted(set(required))},
        "original_hashes": {relative: sha(original / relative) for relative in sorted(original_sources)},
        "reference_manifest": "../study_2026-10-08/data/freeze.json (contains original reference digest; no reference labels read by preparation)",
    }
    save(root / "protocol_frozen.json", protocol)
    print(json.dumps(protocol["counts"]))
    return protocol


def committed_response(root, job, protocol, runner, config=None):
    folder = root / "runs/phase1" / job["id"]
    if not (folder / "meta.json").exists():
        raise RuntimeError("Phase 1 not finished: " + job["id"])
    meta = read(folder / "meta.json")
    response = (folder / "response.txt").read_bytes().decode("utf-8")
    prompt_path = (root / "manifests" / job["prompt_file"]).resolve()
    prompt = runner.WRAPPER + prompt_path.read_text(encoding="utf-8")
    if meta.get("id") != job["id"] or meta.get("model_requested") != job["model"]:
        raise ValueError("Committed model/job differs: " + job["id"])
    if digest(response) != meta.get("response_sha256") or digest(prompt) != meta.get("prompt_sha256"):
        raise ValueError("Committed response/prompt changed: " + job["id"])
    if (folder / "input.txt").read_text(encoding="utf-8") != prompt:
        raise ValueError("Committed sent input changed: " + job["id"])
    if config and config["prompt_hashes"][job["id"]] != sha(prompt_path):
        raise ValueError("Committed prompt differs from run configuration")
    payload, _, _ = runner.build_request(job, protocol["model_settings"][job["model"]], prompt)
    request_hash = hashlib.sha256(runner.encode(payload)).hexdigest()
    if meta.get("request_sha256") != request_hash or (config and config["request_hashes"].get(job["id"]) != request_hash):
        raise ValueError("Committed request differs from frozen model settings")
    if (folder / "request.json").exists() and sha(folder / "request.json") != request_hash:
        raise ValueError("Committed serialized request changed")
    if meta.get("raw_response_sha256") is not None and sha(folder / "raw_response.json") != meta["raw_response_sha256"]:
        raise ValueError("Committed raw API response changed")
    valid_format, _ = runner.validate_answer(response, job["item_ids"])
    inference_ok = meta.get("returncode") == 0 and meta.get("completed") is True and meta.get("tool_event_count") == 0
    valid = bool(meta.get("valid") and valid_format and inference_ok)
    if bool(meta.get("valid")) != valid:
        raise ValueError("Committed validity contradicts independent validation")
    return meta, response, valid


def phase2(root=ROOT):
    protocol = verify_freeze(root)
    original = root.parent / "study_2026-10-08"
    runner = load_module(root / "runner.py", "extension_runner_preparation")
    first = read(root / "manifests/phase1.json")
    config = read(root / "runs/phase1/run_config.json")
    if config.get("runner_sha256") != protocol["hashes"]["runner.py"] or config.get("manifest_sha256") != sha(root / "manifests/phase1.json"):
        raise ValueError("Phase 1 run differs from freeze")
    if config.get("wrapper_sha256") != digest(runner.WRAPPER):
        raise ValueError("Phase 1 wrapper differs")
    if config.get("models") != protocol["model_settings"]:
        raise ValueError("Phase 1 model settings differ")
    # Complete all S/C/E before constructing any second-pass prompt.
    responses = {j["id"]: committed_response(root, j, protocol, runner, config) for j in first["jobs"]}
    symbolic = read(root / "design/verifier_analyses.json")
    jobs, blocked = [], []
    for planned in read(root / "design/revision_plan.json"):
        meta, response, valid = responses[planned["dependency"]]
        if not valid:
            blocked.append({**planned, "reason": "Initial S inference invalid; prespecified dependent failure"})
            continue
        text = revision_prompt(planned, response, symbolic, original)
        destination = root / "prompts" / (planned["id"] + ".txt")
        save_bytes(destination, text.encode("utf-8"))
        jobs.append({**{k: v for k, v in planned.items() if k != "items"},
                     "model": protocol["models"][planned["model_key"]],
                     "prompt_file": "../prompts/" + destination.name,
                     "item_ids": [x["id"] for x in planned["items"]],
                     "dependency_response_sha256": meta["response_sha256"]})
    save(root / "manifests/phase2.json", {"models": protocol["model_settings"], "jobs": jobs})
    save(root / "manifests/phase2_blocked.json", blocked)
    phase_paths = [root / "manifests/phase2.json", root / "manifests/phase2_blocked.json"]
    phase_paths += [(root / "manifests" / j["prompt_file"]).resolve() for j in jobs]
    save(root / "design/phase2_freeze.json", {"protocol_sha256": sha(root / "protocol_frozen.json"),
         "hashes": {p.relative_to(root).as_posix(): sha(p) for p in phase_paths},
         "dependencies": {p["dependency"]: responses[p["dependency"]][0]["response_sha256"]
                          for p in read(root / "design/revision_plan.json")}})
    print(json.dumps({"phase2_jobs": len(jobs), "blocked_batches": len(blocked), "planned_revisions": len(jobs) + len(blocked)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "phase2", "verify"))
    args = parser.parse_args()
    {"freeze": freeze, "phase2": phase2, "verify": verify_freeze}[args.action]()
