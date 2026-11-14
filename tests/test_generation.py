"""Real Transformers configuration regression checks, CPU only, no downloads.

Optional when running the standard-library suite before dependency installation.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contract_experiment.runtime import (LocalModel, effective_greedy_config,
                                         greedy_overrides)
from contract_experiment.recovery import verified_compressor_details
from contract_experiment.storage import write_json

try:
    import torch
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
except ImportError:
    GenerationConfig = None


@unittest.skipIf(GenerationConfig is None, "Install pinned dependencies for the CPU generation-config regression")
class GenerationTests(unittest.TestCase):
    def adapter(self, version):
        class ConfigOnly(GenerationMixin):
            pass
        adapter = ConfigOnly()
        adapter.generation_config = GenerationConfig(do_sample=True, temperature=0.6,
            top_k=20, top_p=0.95, bos_token_id=151643, transformers_version=version)
        return adapter

    def test_qwen3_model_defaults_cannot_reenable_sampling(self):
        adapter = self.adapter("4.51.0")
        requested = GenerationConfig(do_sample=False, num_beams=1)
        legacy, _ = adapter._prepare_generation_config(requested, max_new_tokens=32)
        self.assertTrue(legacy.do_sample)  # Reproduce the actual library bug trigger.
        fixed = effective_greedy_config(adapter, requested, 32)
        actual, _ = adapter._prepare_generation_config(fixed, **greedy_overrides(), max_new_tokens=32)
        self.assertFalse(actual.do_sample)
        self.assertEqual(actual.num_beams, 1)
        self.assertEqual(fixed.to_dict(), actual.to_dict())

    def test_legacy_qwen25_compression_config_is_unchanged(self):
        adapter = self.adapter("4.37.0")
        requested = GenerationConfig(do_sample=False, num_beams=1)
        legacy, _ = adapter._prepare_generation_config(requested, max_new_tokens=4096)
        fixed = effective_greedy_config(adapter, requested, 4096)
        self.assertFalse(legacy.do_sample)
        self.assertEqual(legacy.to_dict(), fixed.to_dict())

    def test_recovery_rejects_a_sampled_legacy_compressor(self):
        env = {"packages": {"transformers": "4.57.1"}}
        models = {"models": {"compressor": {"repo_id": "fixture/model", "revision": "a" * 40}}}
        details = {**env, "role": "compressor", "model_revision": "a" * 40,
                   "generation_config": GenerationConfig(do_sample=False).to_dict()}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            write_json(source / "runtime_compressor.json", details)
            with patch("contract_experiment.recovery.environment", return_value=env), patch.object(
                    GenerationConfig, "from_pretrained", return_value=self.adapter("4.51.0").generation_config):
                with self.assertRaisesRegex(ValueError, "cannot be reused"):
                    verified_compressor_details(source, {"generation": {"compressor_max_new_tokens": 4096}}, models)

    def test_production_generate_passes_explicit_overrides(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "prompt"

            def __call__(self, *args, **kwargs):
                return Inputs(input_ids=torch.tensor([[8, 9]]))

            def decode(self, ids, skip_special_tokens):
                return "Entailment"

        adapter = self.adapter("4.51.0")
        observed = {}

        def generate(**kwargs):
            observed.update(kwargs)
            effective, _ = adapter._prepare_generation_config(
                kwargs.pop("generation_config"), **kwargs)
            self.assertFalse(effective.do_sample)
            self.assertEqual(effective.num_beams, 1)
            return torch.tensor([[8, 9, 1]])

        adapter.generate = generate
        local = LocalModel.__new__(LocalModel)  # Never instantiate/load a real model.
        local.torch, local.model, local.tokenizer = torch, adapter, Tokenizer()
        local.config = {"generation": {"judge_max_new_tokens": 32, "stop_strings": []}}
        local.role, local.device, local.context_limit, local.eos_ids = "judge", "cpu", 100, {1}
        local.generation_config = effective_greedy_config(adapter, GenerationConfig(), 32)
        with patch.object(torch.cuda, "synchronize"):
            output = local.generate([{"role": "user", "content": "data"}])
        self.assertEqual({key: observed[key] for key in greedy_overrides()}, greedy_overrides())
        self.assertEqual(output["stop_reason"], "eos")
        self.assertEqual(output["decoding_policy"], "explicit_greedy_v1")


if __name__ == "__main__":
    unittest.main()
