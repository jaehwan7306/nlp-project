"""Recovery journal isolation checks with synthetic records, never real models."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contract_experiment import CONDITIONS
from contract_experiment.recovery import reuse_compression
from contract_experiment.storage import (append_jsonl, read_json, read_jsonl,
                                         unique_index, write_json)


class RecoveryTests(unittest.TestCase):
    def test_only_contexts_are_reused_and_old_judgments_stay_separate(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            source, target = Path(temporary) / "source", Path(temporary) / "target"
            source.mkdir()
            target.mkdir()
            identity = {"config": "c", "models": "m", "prepared": "d", "prompts": "p"}
            manifest = {"split": "dev", "document_count": 1, "full_split_count": 1,
                        "identity": {**identity, "code": "old"}}
            new_manifest = {**manifest, "identity": {**identity, "code": "new"}}
            rows = [{"document": "dev:1", "condition": condition, "text": condition,
                     "compression_seconds": 12.0} for condition in CONDITIONS]
            for row in rows:
                append_jsonl(source / "contexts.jsonl", row)
            append_jsonl(target / "contexts.jsonl", rows[0])
            append_jsonl(source / "predictions.jsonl", {"prediction": "old sampled judgment"})
            for folder in (source, target):
                write_json(folder / "environment.json", {})
            write_json(source / "runtime_compressor.json", {"decoding": "old metadata"})
            source_before = (source / "contexts.jsonl").read_bytes()
            dataset = {"documents": [{"key": "dev:1"}]}
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("contract_experiment.recovery.verified_source", return_value=(source, manifest, {}, {}, dataset)))
                stack.enter_context(patch("contract_experiment.recovery.load_run", return_value=(target, new_manifest, {}, {}, dataset)))
                stack.enter_context(patch("contract_experiment.recovery.environment", return_value={}))
                stack.enter_context(patch("contract_experiment.recovery.validate_contexts", return_value=unique_index(rows, ("document", "condition"))))
                stack.enter_context(patch("contract_experiment.recovery.verified_compressor_details", return_value=({}, {"effective_config_identical": True})))
                reuse_compression(source, target, check_only=True)
                self.assertEqual(read_jsonl(target / "contexts.jsonl"), rows[:1])
                self.assertFalse((target / "compression_reuse.json").exists())
                reuse_compression(source, target)
                self.assertEqual(read_jsonl(target / "contexts.jsonl"), rows)
                self.assertFalse((target / "predictions.jsonl").exists())
                proof = read_json(target / "compression_reuse.json")
                self.assertEqual(proof["judgments_imported"], 0)
                self.assertEqual(proof["compressed_contexts"], 4)
                self.assertEqual((source / "contexts.jsonl").read_bytes(), source_before)
                append_jsonl(target / "predictions.jsonl", {"prediction": "existing"})
                with self.assertRaisesRegex(ValueError, "Target has judgments"):
                    reuse_compression(source, target)


if __name__ == "__main__":
    unittest.main()
