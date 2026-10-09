"""Contract and denominator regressions, using arbitrary synthetic labels only."""

from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rhyme_bench.cli import main
from rhyme_bench.contracts import ValidationError, load_records, validate_dataset
from rhyme_bench.scoring import score

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.reference = load_records(ROOT / "data/examples/synthetic_reference.json")
        self.predictions = load_records(ROOT / "data/examples/synthetic_predictions.jsonl")

    def evaluate(self, predictions=None):
        return score(self.reference, self.predictions if predictions is None else predictions,
                     allow_nonhuman_reference=True)

    def test_nonhuman_reference_requires_explicit_override(self):
        with self.assertRaisesRegex(ValidationError, "default scoring requires human"):
            score(self.reference, self.predictions)
        self.assertIn("DIAGNOSTIC ONLY", self.evaluate()["reference"]["warning"])

    def test_adjudicated_metadata_allows_default_scoring(self):
        reference = copy.deepcopy(self.reference)
        for record in reference:
            record["annotation"] = {"status": "human_adjudicated", "annotator_ids": ["a", "b"], "adjudicator_id": "c"}
        # This tests declared metadata, not an assertion of actual human annotation.
        self.assertIsNone(score(reference, self.predictions)["reference"]["warning"])

    def test_denominators_uncertainty_and_negative_type_exclusion(self):
        result = self.evaluate()
        metrics = result["metrics"]
        self.assertEqual(result["counts"]["scored_definite_items"], 3)
        self.assertEqual(result["excluded_uncertain_pair_ids"], ["synthetic-004"])
        self.assertEqual(metrics["detection_accuracy"], {"correct": 2, "total": 3, "value": 2 / 3})
        self.assertEqual(metrics["type_accuracy_on_gold_positives"], {"correct": 1, "total": 2, "value": 0.5})
        self.assertIsNone(result["item_results"][2]["type_correct"])
        self.assertEqual(metrics["feature_micro_on_gold_positives"]["fn"], 2)
        self.assertEqual(metrics["joint_accuracy_on_definite_items"]["correct"], 2)

    def test_predictions_are_aligned_by_id_not_row_order(self):
        self.assertEqual(self.evaluate(), self.evaluate(list(reversed(self.predictions))))

    def test_missing_predictions_stay_in_denominator(self):
        result = self.evaluate([])
        self.assertEqual(result["counts"]["prediction_statuses_on_definite_items"], {"missing": 3})
        self.assertEqual(result["metrics"]["joint_accuracy_on_definite_items"]["value"], 0)
        self.assertEqual(result["metrics"]["type_accuracy_on_gold_positives"]["total"], 2)
        self.assertEqual(result["metrics"]["detection_positive_class"]["fn"], 2)

    def test_errors_and_uncertainty_are_not_correct_on_negative_gold(self):
        for prediction in (
            {"pair_id": "synthetic-003", "status": "error", "rhyme": "uncertain", "rhyme_type": None, "features": [], "error": "provider timeout"},
            {"pair_id": "synthetic-003", "status": "ok", "rhyme": "uncertain", "rhyme_type": None, "features": []},
        ):
            result = self.evaluate([prediction])
            self.assertEqual(result["metrics"]["detection_accuracy"]["correct"], 0)
            self.assertEqual(result["metrics"]["joint_accuracy_on_definite_items"]["correct"], 0)

    def test_false_positive_is_penalized_on_negative_gold(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[2].update(rhyme="yes", rhyme_type="M", features=["PURE"])
        result = self.evaluate(predictions)
        self.assertEqual(result["metrics"]["detection_positive_class"]["fp"], 1)
        self.assertFalse(result["item_results"][2]["joint_correct"])

    def test_all_abstain_system_has_zero_accuracy_and_coverage(self):
        predictions = [{"pair_id": item["pair_id"], "status": "abstain", "rhyme": "uncertain", "rhyme_type": None, "features": []}
                       for item in self.reference]
        result = self.evaluate(predictions)
        self.assertEqual(result["metrics"]["detection_accuracy"]["value"], 0)
        self.assertEqual(result["metrics"]["answer_coverage"]["value"], 0)
        self.assertEqual(result["metrics"]["type_accuracy_on_gold_positives"]["value"], 0)
        self.assertEqual(result["metrics"]["joint_accuracy_on_definite_items"]["value"], 0)
        self.assertEqual(result["metrics"]["detection_positive_class"]["f1"], 0)
        self.assertIsNone(result["metrics"]["detection_positive_class"]["precision"])
        self.assertEqual(result["counts"]["prediction_statuses_on_definite_items"], {"abstain": 3})

    def test_detection_does_not_mask_wrong_type_or_features(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[0].update(rhyme_type="F2", features=["RICH"])
        result = self.evaluate(predictions)
        self.assertTrue(result["item_results"][0]["detection_correct"])
        self.assertFalse(result["item_results"][0]["type_correct"])
        self.assertFalse(result["item_results"][0]["feature_exact"])
        self.assertFalse(result["item_results"][0]["joint_correct"])

    def test_no_positive_or_definite_reference_has_null_not_perfect_score(self):
        for reference in ([self.reference[2]], [self.reference[3]]):
            result = score(reference, [], allow_nonhuman_reference=True)
            self.assertIsNone(result["metrics"]["type_accuracy_on_gold_positives"]["value"])
            self.assertIsNone(result["metrics"]["feature_micro_on_gold_positives"]["f1"])
        uncertain = score([self.reference[3]], [], allow_nonhuman_reference=True)
        self.assertIsNone(uncertain["metrics"]["joint_accuracy_on_definite_items"]["value"])

    def test_unknown_and_duplicate_prediction_ids_are_rejected(self):
        for predictions in (self.predictions + [self.predictions[0]], [{**self.predictions[0], "pair_id": "typo"}]):
            with self.assertRaises(ValidationError):
                self.evaluate(predictions)

    def test_legacy_aliases_duplicate_features_and_invalid_status_are_rejected(self):
        mutations = ({"features": ["IMP"]}, {"features": ["PURE", "PURE"]},
                     {"status": "error"}, {"rhyme": "no"}, {"rhyme_type": None},
                     {"unknown_field": "unexpected"})
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                predictions = copy.deepcopy(self.predictions)
                predictions[0].update(mutation)
                with self.assertRaises(ValidationError):
                    self.evaluate(predictions)

    def test_dataset_rejects_duplicates_and_poem_split_leakage(self):
        with self.assertRaisesRegex(ValidationError, "duplicate pair_id"):
            validate_dataset(self.reference + [self.reference[0]])
        record = copy.deepcopy(self.reference[0])
        record.update(pair_id="new-pair", split="test")
        with self.assertRaisesRegex(ValidationError, "multiple splits"):
            validate_dataset(self.reference + [record])

    def test_false_human_metadata_and_missing_source_ids_are_rejected(self):
        for mutation in ({"source_id": ""}, {"annotation": {"status": "human_agreement", "annotator_ids": ["a"], "adjudicator_id": None}}):
            reference = copy.deepcopy(self.reference)
            reference[0].update(mutation)
            with self.assertRaises(ValidationError):
                validate_dataset(reference)

    def test_json_jsonl_bom_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in self.reference), encoding="utf-8-sig")
            self.assertEqual(load_records(path), self.reference)
            path.write_text("{bad json}", encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_records(path)

    def test_cli_output_and_input_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.json"
            reference = str(ROOT / "data/examples/synthetic_reference.json")
            predictions = str(ROOT / "data/examples/synthetic_predictions.jsonl")
            args = ["score", reference, predictions, "--allow-nonhuman-reference", "--output", str(output)]
            self.assertEqual(main(args), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(result["inputs"]["dataset"]["sha256"]), 64)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(args[:-1] + [reference]), 2)
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(["validate-dataset", reference]), 0)
            self.assertTrue(json.loads(stdout.getvalue())["valid"])

    def test_cli_preserves_existing_report_and_hardlinked_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            original = json.dumps(self.reference).encode("utf-8")
            reference.write_bytes(original)
            predictions = ROOT / "data/examples/synthetic_predictions.jsonl"
            args = ["score", str(reference), str(predictions), "--allow-nonhuman-reference", "--output"]
            report = root / "existing-report.json"
            report.write_text("preserve this report", encoding="utf-8")
            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(main(args + [str(report)]), 2)
            self.assertIn("already exists", stderr.getvalue())
            self.assertEqual(report.read_text(encoding="utf-8"), "preserve this report")
            alias = root / "reference-hardlink.json"
            try:
                os.link(reference, alias)
            except OSError as exc:
                self.skipTest(f"filesystem does not support hardlinks: {exc}")
            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(main(args + [str(alias)]), 2)
            self.assertIn("aliases dataset", stderr.getvalue())
            self.assertEqual(reference.read_bytes(), original)
            self.assertEqual(alias.read_bytes(), original)

    def test_invalid_utf8_returns_validation_error_and_cli_exit_two(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            invalid.write_bytes(b"\xff\xfeinvalid utf8")
            with self.assertRaises(ValidationError):
                load_records(invalid)
            commands = (["validate-dataset", str(invalid)],
                        ["score", str(ROOT / "data/examples/synthetic_reference.json"), str(invalid), "--allow-nonhuman-reference"])
            for command in commands:
                with redirect_stderr(io.StringIO()) as stderr:
                    self.assertEqual(main(command), 2)
                self.assertIn("error:", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
