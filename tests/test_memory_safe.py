"""Small CPU attention and provenance checks, no model weights or inference."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contract_experiment import CONDITIONS
from contract_experiment.data import build_units
from contract_experiment.settings import ROOT, environment, identity, load_config, load_run
from contract_experiment.storage import (append_jsonl, digest, read_json, read_jsonl, write_json)
from memory_support.attention import headwise_sdpa, memory_attention, memory_model, policy_spec, runtime_policy_marker
from memory_support.runner import (CHECK_SCHEMA, add_report_note, adopt, check_dev, execute, initialize_test,
                                   verify_new_rows, verify_report, verify_technical_report)

try:
    import torch
    from torch.nn import functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install pinned dependencies for CPU attention equivalence checks")
class AttentionTests(unittest.TestCase):
    def test_full_context_grouped_heads_masks_causal_and_decode(self):
        generator = torch.Generator().manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16):
            for heads, kv_heads in ((4, 2), (4, 4), (4, 1)):
                for qlength, klength, mask_kind in ((7, 7, "causal"), (1, 9, "none"),
                        (3, 9, "bool"), (3, 9, "head_float")):
                    with self.subTest(dtype=dtype, heads=heads, kv=kv_heads, mask=mask_kind):
                        q = torch.randn(2, heads, qlength, 8, generator=generator).to(dtype)
                        k = torch.randn(2, kv_heads, klength, 8, generator=generator).to(dtype)
                        v = torch.randn(2, kv_heads, klength, 6, generator=generator).to(dtype)
                        mask = None
                        if mask_kind == "bool":
                            mask = torch.ones(qlength, klength, dtype=torch.bool)
                            mask[:, -2:] = False
                        elif mask_kind == "head_float":
                            mask = torch.zeros(2, heads, qlength, klength, dtype=dtype)
                            for head in range(heads):
                                mask[:, head, :, head] = -3.0
                        kwargs = {"attn_mask": mask, "is_causal": mask_kind == "causal",
                                  "scale": 0.3, "enable_gqa": heads != kv_heads}
                        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                            expected = functional.scaled_dot_product_attention(q, k, v, **kwargs)
                            actual = headwise_sdpa(functional.scaled_dot_product_attention, q, k, v, **kwargs)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_long_prefill_uses_one_head_without_shortening_tokens_and_restores_function(self):
        native = functional.scaled_dot_product_attention
        calls = []

        def recording(q, k, v, **kwargs):
            calls.append((q.shape, k.shape, kwargs))
            return torch.zeros_like(q)

        stats = {"chunked_calls": 0, "max_query_tokens": 0, "max_key_tokens": 0}
        query = torch.zeros(1, 4, 1025, 2)
        key = torch.zeros(1, 2, 1025, 2)
        with patch.object(functional, "scaled_dot_product_attention", recording):
            with memory_attention(stats):
                output = functional.scaled_dot_product_attention(query, key, key, is_causal=True, enable_gqa=True)
            self.assertIs(functional.scaled_dot_product_attention, recording)
        self.assertIs(functional.scaled_dot_product_attention, native)
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(q[1] == 1 and q[-2] == k[-2] == 1025 for q, k, kwargs in calls))
        self.assertEqual(stats["chunked_calls"], 1)
        with self.assertRaisesRegex(ValueError, "dropout"):
            headwise_sdpa(native, query, key, key, dropout_p=0.1, enable_gqa=True)

    def test_model_output_records_policy_and_exception_restores_sdpa(self):
        class FakeModel:
            def generate(self, messages):
                return {"text": "saved"}

        model = memory_model(FakeModel, policy_spec())()
        result = model.generate([])
        self.assertEqual(result["attention_memory_policy"]["policy_sha256"], digest(policy_spec()))
        native = functional.scaled_dot_product_attention
        with self.assertRaises(RuntimeError):
            with memory_attention({}):
                raise RuntimeError("interrupted")
        self.assertIs(functional.scaled_dot_product_attention, native)

    def test_synthetic_dev_check_and_archived_adoption(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source, target, check = root / "dev", root / "test", root / "check"
            source.mkdir()
            target.mkdir()
            source_text = "A source clause."
            doc = {"key": "dev:1", "units": build_units(source_text, [[0, len(source_text)]])}
            dataset = {"documents": [doc], "hypotheses": {f"h{i}": f"Claim {i}" for i in range(17)}}
            manifest = {"split": "dev", "identity": {"synthetic": "frozen"},
                        "expected_contexts": 5, "expected_judgments": 85,
                        "document_count": 1, "document_keys": ["dev:1"]}
            generated = {"raw_token_ids": [10, 11], "text": "same tokens", "stop_reason": "eos",
                         "rendered_prompt_sha256": "same prompt", "input_tokens": 2000}
            for condition in CONDITIONS:
                append_jsonl(source / "contexts.jsonl", {"document": "dev:1", "condition": condition,
                    "context_tokens": 2000, "text": "source", "generation": dict(generated)})
                for hid in dataset["hypotheses"]:
                    append_jsonl(source / "predictions.jsonl", {"document": "dev:1", "condition": condition,
                        "hypothesis": hid, "generation": dict(generated)})
            runtime = {role: {"decoding_policy": "explicit_greedy_v1", "role": role} for role in ("judge", "compressor")}
            for role, details in runtime.items():
                write_json(source / f"runtime_{role}.json", details)
            write_json(source / "environment.json", {})

            class FakeModel:
                def __init__(self, config, models, role):
                    self.role = role
                    self.details = runtime[role]

                def generate(self, messages):
                    text = "Entailment" if self.role == "judge" else (
                        "[1]" if "JSON" in messages[1]["content"] else "A changed summary.")
                    return {**generated, "raw_token_ids": [12, 13], "text": text,
                            "decoding_policy": "explicit_greedy_v1"}

                def close(self):
                    pass

            loaded = (source, manifest, {}, {}, dataset)
            with patch("memory_support.runner.load_run", return_value=loaded), patch(
                    "memory_support.runner.environment", return_value={}), patch(
                    "contract_experiment.runtime.LocalModel", FakeModel):
                self.assertEqual(check_dev(source, check), 0)
            report = read_json(check / "report.json")
            self.assertEqual(len(report["checks"]), 9)
            self.assertTrue(all(row["passed"] for row in report["checks"]))
            self.assertEqual(report["identical_output_count"], 0)
            marker = runtime_policy_marker(policy_spec())
            write_json(target / "frozen_protocol.json", {"runtime": {
                role: {**details, "attention_memory_policy": marker} for role, details in runtime.items()},
                "memory_execution": {"mode": "from_start", "policy": policy_spec(), "check_report_sha256": digest(report)}})
            with patch("memory_support.runner.environment", return_value={}):
                verify_report(report, manifest, target)
                report["checks"][0]["technical"]["greedy"] = False
                with self.assertRaisesRegex(ValueError, "passing"):
                    verify_technical_report(report, manifest["identity"], runtime)
                report["checks"][0]["technical"]["greedy"] = True
            prefix = {"generation": {"text": "old result"}}
            append_jsonl(target / "predictions.jsonl", prefix)
            with self.assertRaisesRegex(ValueError, "zero judgments"):
                adopt(target, manifest, report)
            (target / "predictions.jsonl").unlink()
            append_jsonl(target / "contexts.jsonl", {"document": "dev:1", "condition": "original", "generation": None})
            record = adopt(target, manifest, report)
            self.assertEqual(record["journal_boundaries"]["predictions.jsonl"], 0)
            self.assertEqual(read_jsonl(target / "predictions.jsonl"), [])
            self.assertTrue((target / "memory_extension_source" / "run_memory_safe.py").exists())
            self.assertEqual(adopt(target, manifest, report), record)
            write_json(target / "results" / "analysis.json", {"metrics": "unchanged"})
            (target / "results" / "report.md").write_text("Synthetic results\n", encoding="utf-8")
            add_report_note(target)
            analysis = read_json(target / "results" / "analysis.json")
            self.assertEqual(analysis["metrics"], "unchanged")
            self.assertEqual(analysis["attention_memory_execution"]["previous_judgments_reused"], 0)
            self.assertEqual(analysis["attention_memory_execution"]["execution_mode"], "from_start")


class MemoryJournalTests(unittest.TestCase):
    def test_preserved_prefix_and_new_policy_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            old = {"document": "test:1", "generation": {"text": "old"}}
            append_jsonl(run / "predictions.jsonl", old)
            policy = {"id": "synthetic policy"}
            record = {"policy": policy, "journal_boundaries": {"predictions.jsonl": 1},
                      "journal_prefix_digests": {"predictions.jsonl": digest([old])}}
            append_jsonl(run / "predictions.jsonl", {"generation": {
                "attention_memory_policy": {"policy_sha256": digest(policy)}}})
            verify_new_rows(run, record)
            append_jsonl(run / "predictions.jsonl", {"generation": {"text": "unrecorded implementation"}})
            with self.assertRaisesRegex(ValueError, "lacks"):
                verify_new_rows(run, record)
            record["journal_prefix_digests"]["predictions.jsonl"] = digest([])
            with self.assertRaisesRegex(ValueError, "Pre-extension"):
                verify_new_rows(run, record)

    def test_fresh_test_uniform_policy_and_native_entry_is_blocked(self):
        from contract_experiment.experiment import judge

        class Tokenizer:
            def encode(self, text, **kwargs):
                return text.split()

        class Model:
            generated = 0

            def __init__(self, config, models, role):
                self.role = role
                self.details = {"role": role, "load_seconds": 0.0, "decoding_policy": "explicit_greedy_v1"}

            def generate(self, messages):
                Model.generated += 1
                text = "Entailment" if self.role == "judge" else (
                    "[1]" if "JSON" in messages[1]["content"] else "New summary.")
                return {"text": text, "raw_text": text, "raw_token_ids": [], "stop_reason": "eos",
                    "cap_reached": False, "dialogue_flag": False, "input_tokens": 10,
                    "generated_tokens": 2, "generation_seconds": 0.0,
                    "rendered_prompt_sha256": "synthetic", "decoding_policy": "explicit_greedy_v1"}

            def close(self):
                pass

        @contextlib.contextmanager
        def cpu_stub(stats):
            yield

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            old_protocol, old_test = root / "old_protocol", root / "old_test"
            protocol, target = root / "memory_protocol", root / "memory_test"
            old_test.mkdir()
            append_jsonl(old_test / "predictions.jsonl", {"text": "old output must never be imported"})
            old_before = (old_test / "predictions.jsonl").read_bytes()
            config_path = root / "config.toml"
            config_path.write_text((ROOT / "config.toml").read_text(encoding="utf-8").replace(
                "bootstrap_repetitions = 10000", "bootstrap_repetitions = 30"), encoding="utf-8")
            config = load_config(config_path)
            models = {"models": {role: {"repo_id": repo, "revision": "a" * 40} for role, repo in config["models"].items()}}
            models_path = root / "models.json"
            write_json(models_path, models)
            prepared = {"counts": {"train": 1, "dev": 1, "test": 1}, "synthetic": True}
            hypotheses = {f"h{i}": f"Claim {i}" for i in range(17)}
            text = "An original contract clause."
            doc = {"key": "test:1", "id": 1, "split": "test", "text": text,
                   "units": build_units(text, [[0, len(text)]]),
                   "annotations": {hid: {"choice": "Entailment", "spans": [0]} for hid in hypotheses}}
            dataset = {"documents": [doc], "hypotheses": hypotheses}
            base_identity = identity(config, prepared, models)
            runtimes = {role: {"role": role, "decoding_policy": "explicit_greedy_v1"} for role in ("judge", "compressor")}
            base = {"identity": base_identity, "environment": environment(), "runtime": runtimes}
            write_json(old_protocol / "protocol.json", base)
            write_json(old_protocol / "test_run.json", {"run_directory": str(old_test)})
            base_before = (old_protocol / "protocol.json").read_bytes()
            roles = [("judge", c) for c in CONDITIONS] + [("compressor", c) for c in CONDITIONS[1:]]
            technical = dict.fromkeys(("same_rendered_prompt", "same_input_tokens", "valid_output_format", "greedy", "memory_policy"), True)
            report = {"schema": CHECK_SCHEMA, "status": "complete", "passed": True,
                "policy": policy_spec(), "identity": base_identity, "environment": environment(), "runtime": runtimes,
                "checks": [{"role": r, "condition": c, "technical": dict(technical), "passed": True,
                            "equal": {"raw_token_ids": False}, "output_identical": False} for r, c in roles]}
            report_path = root / "check.json"
            write_json(report_path, report)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("contract_experiment.data.load_prepared", return_value=(dataset, prepared)))
                stack.enter_context(patch("contract_experiment.experiment.load_prepared", return_value=(dataset, prepared)))
                stack.enter_context(patch("contract_experiment.experiment.load_tokenizer", return_value=Tokenizer()))
                stack.enter_context(patch("contract_experiment.experiment.LocalModel", Model))
                stack.enter_context(patch("memory_support.attention.memory_attention", cpu_stub))
                initialize_test(config_path, root / "prepared", models_path, old_protocol, protocol, target, report_path)
                self.assertFalse((target / "predictions.jsonl").exists())
                self.assertEqual(len(read_jsonl(target / "contexts.jsonl")), 1)
                record = read_json(target / "runtime_memory_policy.json")
                self.assertEqual(record["journal_boundaries"]["predictions.jsonl"], 0)
                self.assertEqual(record["previous_compressed_contexts_reused"], 0)
                with self.assertRaisesRegex(ValueError, "frozen Dev execution"):
                    judge(target, ["original"])
                self.assertEqual(Model.generated, 0)  # Native entry cannot generate even one unmarked row.
                self.assertEqual(execute(["judge", "--run", str(target), "--conditions", "original"]), 0)
                self.assertEqual(execute(["compress", "--run", str(target)]), 0)
                self.assertEqual(execute(["judge", "--run", str(target), "--resume"]), 0)
                before = Model.generated
                self.assertEqual(execute(["judge", "--run", str(target), "--resume"]), 0)
                self.assertEqual(Model.generated, before)
                self.assertEqual(execute(["diagnose", "--run", str(target)]), 0)
                self.assertEqual(execute(["analyze", "--run", str(target)]), 0)
                with self.assertRaisesRegex(ValueError, "already contains"):
                    initialize_test(config_path, root / "prepared", models_path, old_protocol, protocol, target, report_path)
            rows = read_jsonl(target / "predictions.jsonl")
            self.assertEqual(len(rows), 85)
            self.assertTrue(all(r["generation"]["attention_memory_policy"]["policy_sha256"] == digest(policy_spec()) for r in rows))
            self.assertEqual(len(read_jsonl(target / "contexts.jsonl")), 5)
            self.assertEqual((old_test / "predictions.jsonl").read_bytes(), old_before)
            self.assertEqual((old_protocol / "protocol.json").read_bytes(), base_before)
            load_run(target)  # The ordinary immutable-source/snapshot validation remains intact.
            analysis = read_json(target / "results" / "analysis.json")
            self.assertEqual(analysis["attention_memory_execution"]["execution_mode"], "from_start")


if __name__ == "__main__":
    unittest.main()
