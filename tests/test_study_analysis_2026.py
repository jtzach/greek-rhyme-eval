"""Synthetic checks for the automatic-reference study, before outcome analysis."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "study_2026-10-08" / "analysis.py"
SPEC = importlib.util.spec_from_file_location("study_analysis", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def reference(position="M", features=("PURE",)):
    return {"position": position, "features": list(features)}


def prediction(position="M", features=("PURE",)):
    return {"type": position, "features": list(features)}


def row(rep, valid=True, predicted_type="M", features=("PURE",)):
    result = {"split": "main300", "model": "synthetic", "condition": "S", "id": "one",
              "batch": 1, "group": "author_a", "reference_type": "M", "reference_features": ["PURE"],
              "predicted_type": predicted_type, "predicted_features": list(features), "replicate": rep}
    result.update(analysis.score_prediction(reference(), prediction(predicted_type, features), valid))
    return result


class StudyMetricsTests(unittest.TestCase):
    def test_full_set_and_stress_required_for_joint(self):
        exact = analysis.score_prediction(reference(), prediction())
        self.assertEqual(exact["joint"], 1)
        wrong_type = analysis.score_prediction(reference(), prediction("F2"))
        self.assertEqual(wrong_type["feature_exact"], 1)
        self.assertEqual(wrong_type["joint"], 0)
        wrong_features = analysis.score_prediction(reference(), prediction(features=("RICH",)))
        self.assertEqual(wrong_features["stress"], 1)
        self.assertEqual(wrong_features["joint"], 0)

    def test_invalid_outputs_contribute_false_negatives(self):
        result = analysis.score_prediction(reference(features=("PURE", "IDV")), prediction(features=("PURE", "IDV")), valid=False)
        self.assertEqual((result["feature_tp"], result["feature_fp"], result["feature_fn"]), (0, 0, 2))
        self.assertEqual((result["joint"], result["stress"], result["feature_exact"], result["feature_item_f1"]), (0, 0, 0, 0))

    def test_empty_sets_do_not_reward_failed_calls(self):
        ref = reference(features=())
        self.assertEqual(analysis.score_prediction(ref, prediction(features=()))["feature_item_f1"], 1)
        invalid = analysis.score_prediction(ref, None, valid=False)
        self.assertEqual(invalid["feature_item_f1"], 0)
        self.assertEqual(invalid["core_joint"], 0)
        self.assertIsNone(analysis.summarize_metrics([invalid])["feature_micro_f1"])

    def test_core_only_excludes_copy_and_mosaic(self):
        ref = reference(features=("PURE", "COPY", "MOSAIC"))
        result = analysis.score_prediction(ref, prediction())
        self.assertEqual(result["joint"], 0)
        self.assertEqual(result["core_joint"], 1)
        result = analysis.score_prediction(ref, prediction(features=("RICH",)))
        self.assertEqual(result["core_joint"], 0)

    def test_none_with_features_retains_frozen_component_semantics(self):
        result = analysis.score_prediction(reference(), prediction("NONE"))
        self.assertEqual(result["none_with_nonempty_features"], 1)
        self.assertEqual(result["feature_exact"], 1)
        self.assertEqual(result["stress"], 0)
        self.assertEqual(result["joint"], 0)

    def test_feature_order_is_irrelevant_without_inferred_pure(self):
        result = analysis.score_prediction(reference(features=("PURE", "IDV")), prediction(features=("IDV", "PURE")))
        self.assertEqual(result["joint"], 1)
        self.assertEqual(analysis.score_prediction(reference(), prediction(features=()))["feature_exact"], 0)

    def test_macro_item_and_micro_f1_have_different_denominators(self):
        rows = [analysis.score_prediction(reference(), prediction()),
                analysis.score_prediction(reference(features=("RICH", "IDV", "MOSAIC")), prediction(features=()))]
        summary = analysis.summarize_metrics(rows)
        self.assertEqual(summary["feature_item_f1"], .5)
        self.assertEqual(summary["feature_micro_f1"], .4)
        self.assertEqual(summary["n"], 2)

    def test_repeat_average_and_valid_stability(self):
        result = analysis.mean_replicates([row(1), row(2, predicted_type="F2")])[0]
        self.assertEqual(result["joint"], .5)
        self.assertEqual(result["stable_valid_label_set"], 0)
        self.assertEqual(result["both_replicates_valid"], 1)
        failed = analysis.mean_replicates([row(1, False), row(2, False)])[0]
        self.assertEqual(failed["stable_valid_label_set"], 0)
        self.assertEqual(failed["joint"], 0)

    def test_missing_or_duplicated_repeat_is_rejected(self):
        with self.assertRaises(ValueError):
            analysis.mean_replicates([row(1)])
        with self.assertRaises(ValueError):
            analysis.mean_replicates([row(1), row(1)])


class StudyBootstrapTests(unittest.TestCase):
    def test_constant_cluster_estimate_has_zero_width_interval(self):
        result = analysis.cluster_bootstrap([.5] * 30, list(range(30)), 100, 7)
        self.assertEqual((result["estimate"], result["lower"], result["upper"]), (.5, .5, .5))

    def test_unequal_groups_preserve_item_weighted_estimate(self):
        result = analysis.cluster_bootstrap([1, 1, 1, 0], ["big", "big", "big", "small"], 500, 7)
        self.assertEqual(result["estimate"], .75)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["items"], 4)
        self.assertEqual(result, analysis.cluster_bootstrap([1, 1, 1, 0], ["big", "big", "big", "small"], 500, 7))

    def test_paired_identical_conditions_have_exactly_zero_gain(self):
        lhs = [{"id": str(i), "batch": i // 2, "joint": (i % 2) / 2} for i in range(10)]
        result = analysis.paired_contrast(lhs, list(reversed(lhs)), resamples=500, seed=9)
        self.assertEqual((result["estimate"], result["lower"], result["upper"]), (0, 0, 0))

    def test_pairing_requires_matching_ids_and_groups(self):
        lhs = [{"id": "a", "batch": 1, "joint": 1}]
        with self.assertRaises(ValueError):
            analysis.paired_contrast(lhs, [{"id": "b", "batch": 1, "joint": 1}], resamples=10)
        with self.assertRaises(ValueError):
            analysis.paired_contrast(lhs, [{"id": "a", "batch": 2, "joint": 1}], resamples=10)

    def test_complete_denominator_for_invalid_item(self):
        valid, invalid = row(1), row(2, False)
        mean = analysis.mean_replicates([valid, invalid])[0]
        self.assertEqual(mean["joint"], .5)
        self.assertEqual(mean["feature_tp"], .5)
        self.assertEqual(mean["feature_fn"], .5)


class StudyCollectionTests(unittest.TestCase):
    def setUp(self):
        parent = PATH.parent / "analysis" / "synthetic_validation" / "temporary"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="offline_test_", dir=parent)
        self.root = Path(self.temp.name).resolve()
        self.assertIn(parent.resolve(), self.root.parents)
        (self.root / "runner.py").write_bytes((PATH.parent / "runner.py").read_bytes())
        self.runner = analysis._load_runner(self.root)
        jobs = []
        for split, item_id in (("main300", "pair_001"), ("legacy40", "legacy_001")):
            analysis.write_json(self.root / "data/references" / (split + ".json"),
                [{"id": item_id, "reference": reference(), "provenance": {"grouping_key": "fixture_author"}}])
            for rep in (1, 2):
                jid = f"synthetic_S_r{rep}_{split}"
                prompt_path = self.root / "prompts" / (jid + ".txt")
                prompt_path.parent.mkdir(exist_ok=True)
                prompt_path.write_text("Synthetic fixture only: " + item_id, encoding="utf-8")
                jobs.append({"id": jid, "item_ids": [item_id], "model": "test_model",
                             "model_key": "synthetic", "condition": "S", "replicate": rep,
                             "batch": 1, "split": split, "prompt_file": "../prompts/" + prompt_path.name})
        self.jobs = jobs
        analysis.write_json(self.root / "manifests/phase1.json", {"reasoning_effort": "high", "jobs": jobs})
        analysis.write_json(self.root / "manifests/phase2.json", {"reasoning_effort": "high", "jobs": []})
        analysis.write_json(self.root / "manifests/phase2_blocked.json", [])
        analysis.write_json(self.root / "design/revision_plan.json", [])
        self.protocol = {"models": {"synthetic": "test_model"}, "reasoning_effort": "high",
                         "hashes": {"runner.py": analysis.sha(self.root / "runner.py")},
                         "counts": {"main_prediction_records": 2, "legacy_prediction_records": 2}}

    def tearDown(self):
        self.temp.cleanup()

    def completed_fixture(self):
        hashes = {j["id"]: analysis.sha(self.root / "manifests" / j["prompt_file"]) for j in self.jobs}
        analysis.write_json(self.root / "runs/phase1/run_config.json", {
            "runner_sha256": self.protocol["hashes"]["runner.py"], "reasoning_effort": "high",
            "manifest_sha256": analysis.sha(self.root / "manifests/phase1.json"),
            "wrapper_sha256": hashlib.sha256(self.runner.WRAPPER.encode()).hexdigest(), "prompt_hashes": hashes})
        for job in self.jobs:
            folder = self.root / "runs/phase1" / job["id"]
            folder.mkdir(parents=True)
            response = json.dumps({"predictions": [{"id": job["item_ids"][0], **prediction()}]})
            prompt = self.runner.WRAPPER + (self.root / "manifests" / job["prompt_file"]).read_text(encoding="utf-8")
            (folder / "response.txt").write_text(response, encoding="utf-8")
            (folder / "input.txt").write_text(prompt, encoding="utf-8")
            meta = {"id": job["id"], "model_requested": "test_model", "reasoning_effort_requested": "high", "valid": True,
                    "returncode": 0, "completed": True, "tool_event_count": 0,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "elapsed_seconds": 2.0, "usage": {"input_tokens": 100, "output_tokens": 10}}
            analysis.write_json(folder / "meta.json", meta)

    def test_missing_jobs_stay_in_full_denominator(self):
        rows, jobs, missing = analysis.collect_predictions(self.root, self.protocol, allow_partial=True)
        self.assertEqual((len(rows), len(jobs), len(missing)), (4, 4, 4))
        self.assertTrue(all(row["joint"] == 0 and row["feature_fn"] == 1 for row in rows))
        with self.assertRaises(RuntimeError):
            analysis.collect_predictions(self.root, self.protocol)

    def test_committed_predictions_are_read_without_inference(self):
        self.completed_fixture()
        rows, jobs, missing = analysis.collect_predictions(self.root, self.protocol)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row["joint"] for row in rows), 4)
        self.assertEqual(missing, [])
        self.assertEqual(analysis.summarize_cost(jobs)["input_tokens"], 400)

    def test_response_mutation_is_not_silently_scored(self):
        self.completed_fixture()
        folder = self.root / "runs/phase1" / self.jobs[0]["id"]
        (folder / "response.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Response hash differs"):
            analysis.collect_predictions(self.root, self.protocol)

    def test_phase2_cannot_reassign_frozen_condition(self):
        plan = {"id": "synthetic_R_r1", "model_key": "synthetic", "condition": "R", "replicate": 1,
                "batch": 1, "split": "main300", "dependency": self.jobs[0]["id"], "items": [{"id": "pair_001"}]}
        analysis.write_json(self.root / "design/revision_plan.json", [plan])
        phase2 = {**plan, "condition": "W", "item_ids": ["pair_001"], "model": "test_model"}
        analysis.write_json(self.root / "manifests/phase2.json", {"jobs": [phase2]})
        with self.assertRaisesRegex(ValueError, "changed planned metadata"):
            analysis.planned_jobs(self.root, self.protocol["models"])

    def test_unknown_phase2_job_is_rejected(self):
        analysis.write_json(self.root / "manifests/phase2.json", {"jobs": [{"id": "not_planned"}]})
        with self.assertRaisesRegex(ValueError, "Unexpected/overlapping"):
            analysis.planned_jobs(self.root, self.protocol["models"])

    def test_missing_retry_usage_and_unknown_latency_are_reported(self):
        folder = self.root / "cost_fixture"
        analysis.write_json(folder / "attempt_1/meta.json", {"elapsed_seconds": 2.0, "usage": {"input_tokens": 100}})
        interrupted = {"elapsed_seconds": 0, "usage": None, "interrupted_unknown_outcome": True}
        analysis.write_json(folder / "attempt_2/recovery_failure.json", interrupted)
        record = analysis.cost_record(folder, interrupted)
        self.assertEqual(record["all_attempt_seconds"], 2)
        self.assertTrue(record["incomplete_usage"])
        self.assertTrue(record["incomplete_latency"])
        summary = analysis.summarize_cost([{**record, "status": "failure"}])
        self.assertEqual(summary["attempts_without_usage"], 1)
        self.assertEqual(summary["jobs_with_incomplete_latency"], 1)
        self.assertIsNone(summary["median_call_batch_seconds"])

    def test_pipeline_latency_sums_shared_initial_and_second_pass(self):
        first = {"job_id": "S", "all_attempt_seconds": 2, "incomplete_latency": False}
        second = {"job_id": "V", "dependency": "S", "all_attempt_seconds": 3, "incomplete_latency": False}
        summary = analysis.pipeline_latency([second], [first, second])
        self.assertEqual(summary["mean_end_to_end_batch_seconds"], 5)
        self.assertEqual(summary["known_elapsed_batches"], 1)


if __name__ == "__main__":
    unittest.main()
