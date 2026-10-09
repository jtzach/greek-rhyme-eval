"""Independent transport/checkpoint regressions; subprocess is always mocked."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "study_runner_under_test", ROOT / "study_2026-10-08" / "runner.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
MODEL = "gpt-6-astra"
GOOD = json.dumps({"predictions": [{"id": "synthetic", "type": "M", "features": ["PURE"]}]})


class StudyRunnerTest(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="runner-regression-", dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "output"
        self.manifest = self.root / "manifest.json"
        self.prompt = "Synthetic format test only."
        (self.root / "prompt.txt").write_text(self.prompt, encoding="utf-8")
        self.write_manifest(["synthetic"])
        self.exec_calls = []

    def write_manifest(self, ids):
        self.manifest.write_text(json.dumps({"jobs": [{"id": "job", "model": MODEL,
                                  "prompt_file": "prompt.txt", "item_ids": ids}]}), encoding="utf-8")

    def invoke(self, response=GOOD, returncode=0, completed=True, tool=False):
        def fake_run(command, **kwargs):
            if command[-1] == "--version":
                return SimpleNamespace(stdout="synthetic-cli-1\n", stderr="", returncode=0)
            self.exec_calls.append((command, kwargs))
            events = [{"type": "thread.started", "thread_id": "synthetic-thread"},
                      {"type": "item.completed", "item": {"type": "agent_message", "text": response}}]
            if tool:
                events.append({"type": "item.completed", "item": {"type": "command_execution"}})
            if completed:
                events.append({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
            return SimpleNamespace(stdout="\n".join(map(json.dumps, events)), stderr="", returncode=returncode)
        argv = ["runner.py", str(self.manifest), "--output", str(self.out), "--workers", "1"]
        with patch.object(sys, "argv", argv), patch.object(RUNNER.shutil, "which", return_value="fake-codex"), \
             patch.object(RUNNER.subprocess, "run", side_effect=fake_run), \
             patch.object(RUNNER.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            RUNNER.main()
        return json.loads((self.out / "job" / "meta.json").read_text(encoding="utf-8"))

    def existing_attempt(self, response=GOOD, valid=True, returncode=0, completed=True, tool=0):
        directory = self.out / "job" / "attempt_1"
        directory.mkdir(parents=True)
        (directory / "response.txt").write_text(response, encoding="utf-8")
        meta = {"id": "job", "model_requested": MODEL, "attempt": 1,
                "prompt_sha256": RUNNER.digest((RUNNER.WRAPPER + self.prompt).encode()),
                "response_sha256": RUNNER.digest(response.encode()), "returncode": returncode,
                "completed": completed, "tool_event_count": tool,
                "valid": valid, "valid_format": valid, "valid_inference": not tool and completed and returncode == 0,
                "elapsed_seconds": 3, "format_error": None if valid else "Synthetic malformed answer"}
        RUNNER.save(directory / "meta.json", meta)
        (directory / "events.jsonl").write_text("immutable original events", encoding="utf-8")
        return directory

    def test_valid_answer_and_expected_ids(self):
        self.assertEqual(RUNNER.validate_answer(GOOD, ["synthetic"]), (True, None))
        for ids in [[], ["other"], ["synthetic", "other"]]:
            self.assertFalse(RUNNER.validate_answer(GOOD, ids)[0])
        duplicate = json.dumps({"predictions": [json.loads(GOOD)["predictions"][0]] * 2})
        self.assertFalse(RUNNER.validate_answer(duplicate, ["synthetic", "other"])[0])

    def test_invalid_manifest_ids_fail_before_transport(self):
        for ids in [[], ["synthetic", "synthetic"]]:
            self.write_manifest(ids)
            with self.assertRaisesRegex(ValueError, "Empty or duplicate expected item IDs"):
                self.invoke()
        self.assertEqual(self.exec_calls, [])

    def test_fresh_valid_answer_and_final_checkpoint_resume(self):
        first = self.invoke()
        self.assertTrue(first["valid"])
        self.assertEqual(len(self.exec_calls), 1)
        self.assertIn("--ephemeral", self.exec_calls[0][0])
        self.assertEqual(first, self.invoke())
        self.assertEqual(len(self.exec_calls), 1)

    def test_unknown_interrupted_attempt_preserves_evidence_without_resampling(self):
        directory = self.out / "job" / "attempt_1"
        directory.mkdir(parents=True)
        evidence = directory / "response.txt"
        evidence.write_text(GOOD, encoding="utf-8")
        meta = self.invoke()
        self.assertEqual(self.exec_calls, [])
        self.assertFalse(meta["valid"])
        self.assertTrue(meta["interrupted_unknown_outcome"])
        self.assertEqual(evidence.read_text(encoding="utf-8"), GOOD)
        self.assertFalse((self.out / "job" / "attempt_2").exists())

    def test_completed_record_recovered_without_resampling(self):
        directory = self.existing_attempt()
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        meta = self.invoke()
        self.assertEqual(self.exec_calls, [])
        self.assertTrue(meta["valid"])
        self.assertEqual(meta["attempt_count"], 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_completed_malformed_answer_not_retried_fresh_or_recovered(self):
        malformed = "not JSON"
        meta = self.invoke(response=malformed)
        self.assertEqual(len(self.exec_calls), 1)
        self.assertFalse(meta["valid"])
        self.assertTrue(meta["valid_inference"])
        # Simulate interruption after committed attempt and before final checkpoint.
        (self.out / "job" / "meta.json").unlink()
        meta = self.invoke()
        self.assertEqual(len(self.exec_calls), 1)
        self.assertFalse(meta["valid"])
        self.assertEqual((self.out / "job" / "response.txt").read_text(encoding="utf-8"), malformed)

    def test_failed_transport_completed_event_retried_on_resume(self):
        directory = self.existing_attempt(returncode=1, completed=True, valid=False)
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        meta = self.invoke()
        self.assertEqual(len(self.exec_calls), 1)
        self.assertTrue(meta["valid"])
        self.assertEqual(meta["attempt_count"], 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_tool_use_never_retried_or_credited(self):
        meta = self.invoke(tool=True, returncode=1)
        self.assertEqual(len(self.exec_calls), 1)
        self.assertFalse(meta["valid"])
        self.assertEqual(meta["tool_event_count"], 1)

    def test_mutated_saved_response_rejected_without_transport(self):
        directory = self.existing_attempt()
        (directory / "response.txt").write_text("mutated", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Attempt response changed"):
            self.invoke()
        self.assertEqual(self.exec_calls, [])


if __name__ == "__main__":
    unittest.main()
