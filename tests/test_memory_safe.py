"""Small CPU attention and provenance checks, no model weights or inference."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contract_experiment import CONDITIONS
from contract_experiment.storage import (append_jsonl, digest, read_json, read_jsonl, write_json)
from memory_support.attention import headwise_sdpa, memory_attention, memory_model, policy_spec
from memory_support.runner import add_report_note, adopt, check_dev, verify_new_rows, verify_report

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
            doc = {"key": "dev:1", "units": [{"id": 1, "text": "A source clause."}]}
            dataset = {"documents": [doc], "hypotheses": {f"h{i}": f"Claim {i}" for i in range(17)}}
            manifest = {"split": "dev", "identity": {"synthetic": "frozen"},
                        "expected_contexts": 5, "expected_judgments": 85}
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
            write_json(target / "frozen_protocol.json", {"runtime": runtime})

            class FakeModel:
                def __init__(self, config, models, role):
                    self.details = runtime[role]

                def generate(self, messages):
                    return dict(generated)

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
            with patch("memory_support.runner.environment", return_value={}):
                verify_report(report, manifest, target)
                report["checks"][0]["equal"]["raw_token_ids"] = False
                with self.assertRaisesRegex(ValueError, "passing"):
                    verify_report(report, manifest, target)
                report["checks"][0]["equal"]["raw_token_ids"] = True
            prefix = {"generation": {"text": "old result"}}
            append_jsonl(target / "predictions.jsonl", prefix)
            record = adopt(target, manifest, report)
            self.assertEqual(record["journal_boundaries"]["predictions.jsonl"], 1)
            self.assertEqual(read_jsonl(target / "predictions.jsonl"), [prefix])
            self.assertTrue((target / "memory_extension_source" / "run_memory_safe.py").exists())
            self.assertEqual(adopt(target, manifest, report), record)
            write_json(target / "results" / "analysis.json", {"metrics": "unchanged"})
            (target / "results" / "report.md").write_text("Synthetic results\n", encoding="utf-8")
            add_report_note(target)
            analysis = read_json(target / "results" / "analysis.json")
            self.assertEqual(analysis["metrics"], "unchanged")
            self.assertEqual(analysis["attention_memory_execution"]["preserved_journal_prefix"]["predictions.jsonl"], 1)


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


if __name__ == "__main__":
    unittest.main()
