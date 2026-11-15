"""Q17 prompt isolation, paired effects and synthetic end-to-end reuse checks."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contract_experiment import CONDITIONS, LABELS
from contract_experiment import experiment
from contract_experiment.data import build_units
from contract_experiment.prompts import EXTRACTIVE_TASK, PRIORITY, compression_messages as main_messages
from contract_experiment.settings import ROOT, environment, identity, load_config, load_run
from contract_experiment.storage import digest, file_hash, read_json, read_jsonl, write_json
from memory_support.attention import memory_model, policy_spec, runtime_policy_marker
from question_support import runner
from question_support.comparison import paired_bootstrap
from question_support.prompts import compression_messages, hypothesis_block
from question_support.state import inspect_main, load_question_run


def hypotheses():
    return {f"h{i}": f"Unique hypothesis number {i}." for i in range(17)}


class PromptTests(unittest.TestCase):
    def test_only_the_shared_question_block_changes_and_no_gold_is_accessible(self):
        hs = hypotheses()
        units = build_units("A must act. Unless excused.", [[0, 11], [12, 26]])
        block = hypothesis_block(hs)
        for condition in CONDITIONS[1:]:
            baseline = main_messages(condition, units)
            aware = compression_messages(condition, units, hs)
            self.assertEqual(baseline[0], aware[0])
            self.assertEqual(aware[1]["content"].replace("\n\n" + block, "", 1), baseline[1]["content"])
            self.assertEqual(aware[1]["content"].count("<evaluation_hypotheses>"), 1)
            for text in hs.values():
                self.assertEqual(aware[1]["content"].count(text), 1)
            for forbidden in ("GOLD_SENTINEL", "Entailment", "Contradiction", "NotMentioned", "token budget", "target ratio"):
                self.assertNotIn(forbidden, str(aware))
        for mode in ("generative", "extractive"):
            general = compression_messages(mode + "_general", units, hs)
            priority = compression_messages(mode + "_priority", units, hs)
            self.assertEqual(general[1]["content"], priority[1]["content"].replace("\n" + PRIORITY, "", 1))
        with self.assertRaises(ValueError):
            hypothesis_block({"only": "one"})

    def test_document_paired_bootstrap_recomputes_f1_and_rejects_mismatched_gold(self):
        old, new = [], []
        for di, d in enumerate(("test:1", "test:2")):
            for hi, label in enumerate(LABELS):
                row = {"document": d, "condition": "method", "hypothesis": str(hi), "gold": label,
                       "prediction": label if di == 0 else None}
                old.append(row)
                new.append({**row, "prediction": label})
        result = paired_bootstrap(old, new, ["test:1", "test:2"], ["method"], 100, 42)
        self.assertEqual(result["method"]["accuracy"], [0.0, 1.0])
        self.assertEqual(result["method"]["macro_f1"], [0.0, 1.0])
        new[0]["gold"] = "NotMentioned"
        with self.assertRaisesRegex(ValueError, "same document"):
            paired_bootstrap(old, new, ["test:1", "test:2"], ["method"], 10, 42)


class Tokenizer:
    chat_template = "fixture-template"

    def encode(self, text, **kwargs):
        return text.split()

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)


class Model:
    calls = []

    @staticmethod
    def details_for(role):
        return {"role": role, "fixture_only": True, "generation_config": {"do_sample": False, "num_beams": 1},
                "decoding_policy": "explicit_greedy_v1"}

    def __init__(self, config, lock, role):
        self.role = role
        self.details = {**self.details_for(role), "load_seconds": 0.0}

    def generate(self, messages):
        self.calls.append((self.role, messages))
        questions = "<evaluation_hypotheses>" in messages[1]["content"]
        if self.role == "compressor":
            if EXTRACTIVE_TASK in messages[1]["content"]:
                # Main has one deliberate format failure; Q17 valid output must
                # not silently remove those original wrong rows from the effect.
                text = "[999]" if not questions and PRIORITY not in messages[1]["content"] else "[1]"
            else:
                text = "A question-aware summary." if questions else "A Main summary."
        else:
            text = "Entailment"
        return {"text": text, "raw_text": text, "raw_token_ids": [1, 2], "stop_reason": "eos",
            "cap_reached": False, "dialogue_flag": False, "input_tokens": 10,
            "generated_tokens": 2, "generation_seconds": 0.0, "rendered_prompt_sha256": digest(messages),
            "decoding_policy": "explicit_greedy_v1"}

    def close(self):
        pass


@contextlib.contextmanager
def cpu_attention(stats):
    yield


class WorkflowTests(unittest.TestCase):
    def test_shared_q17_compression_original_reuse_runtime_control_and_effects(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), contextlib.ExitStack() as stack:
            root = Path(temporary)
            config_path, model_path = root / "config.toml", root / "models.json"
            config_path.write_text((ROOT / "config.toml").read_text(encoding="utf-8").replace(
                "bootstrap_repetitions = 10000", "bootstrap_repetitions = 30"), encoding="utf-8")
            config = load_config(config_path)
            models = {"models": {role: {"repo_id": repo, "revision": "a" * 40} for role, repo in config["models"].items()}}
            write_json(model_path, models)
            hs = hypotheses()
            docs = []
            for split, di in (("dev", 2), ("test", 1)):
                text = f"Source contract for {split}."
                docs.append({"key": f"{split}:{di}", "id": di, "split": split, "text": text,
                    "units": build_units(text, [[0, len(text)]]),
                    "annotations": {h: {"choice": LABELS[i % 3], "spans": [0], "note": "GOLD_SENTINEL"} for i, h in enumerate(hs)}})
            dataset, prepared = {"documents": docs, "hypotheses": hs}, {"counts": {"dev": 1, "test": 1}}
            stack.enter_context(patch("contract_experiment.experiment.load_prepared", return_value=(dataset, prepared)))
            stack.enter_context(patch("contract_experiment.experiment.load_tokenizer", return_value=Tokenizer()))
            stack.enter_context(patch("question_support.state.load_tokenizer", return_value=Tokenizer()))
            stack.enter_context(patch("memory_support.attention.memory_attention", cpu_attention))
            spec = policy_spec()
            marker = runtime_policy_marker(spec)
            protocol, main, dev = root / "main_protocol", root / "main", root / "dev"
            write_json(protocol / "protocol.json", {"identity": identity(config, prepared, models), "environment": environment(),
                "runtime": {role: {**Model.details_for(role), "attention_memory_policy": marker} for role in ("compressor", "judge")},
                "memory_execution": {"mode": "from_start", "policy": spec}})
            experiment.initialize(config_path, root / "prepared", model_path, "dev", dev)
            experiment.initialize(config_path, root / "prepared", model_path, "test", main, protocol=protocol)
            write_json(main / "runtime_memory_policy.json", {"mode": "from_start", "policy": spec,
                "journal_boundaries": {"contexts.jsonl": 1, "predictions.jsonl": 0}})
            for relative in spec["sources"]:
                target = main / "memory_extension_source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            with patch("contract_experiment.experiment.LocalModel", memory_model(Model, spec, stamp_runtime=True)):
                experiment.judge(main, ["original"])
                experiment.compress(main)
                experiment.judge(main, resume=True)
            experiment.diagnose(main)
            before = {name: file_hash(main / name) for name in ("contexts.jsonl", "predictions.jsonl", "manifest.json", "frozen_protocol.json")}
            inspect_main(main)
            check, test = root / "q17_preflight", root / "q17_test"
            stack.enter_context(patch("contract_experiment.experiment.LocalModel", Model))
            Model.calls = []
            self.assertEqual(runner.preflight(main, dev, check), 0)
            self.assertFalse(read_json(check / "preflight.json")["accuracy_computed"])
            self.assertEqual(sum(role == "compressor" for role, _ in Model.calls), 4)
            self.assertEqual(sum(role == "judge" for role, _ in Model.calls), 85)
            runner.initialize_test(main, check, test)
            self.assertEqual(len(read_jsonl(test / "predictions.jsonl")), 17)
            self.assertTrue(all(r["condition"] == "original" for r in read_jsonl(test / "predictions.jsonl")))
            self.assertEqual(len(read_jsonl(test / "contexts.jsonl")), 1)
            # Ordinary Main code must reject Q17's different prompt/code identity.
            with self.assertRaisesRegex(ValueError, "source changed"):
                load_run(test)
            Model.calls = []
            runner.compress(test)
            runner.judge(test)
            call_count = len(Model.calls)
            runner.compress(test, resume=True)
            runner.judge(test, resume=True)
            self.assertEqual(len(Model.calls), call_count)
            self.assertEqual(sum(role == "compressor" for role, _ in Model.calls), 4)
            self.assertEqual(sum(role == "judge" for role, _ in Model.calls), 68)
            for role, messages in Model.calls:
                self.assertNotIn("GOLD_SENTINEL", str(messages))
                if role == "compressor":
                    self.assertEqual(messages[1]["content"].count("<evaluation_hypotheses>"), 1)
                    self.assertTrue(all(text in messages[1]["content"] for text in hs.values()))
                else:
                    self.assertNotIn("<evaluation_hypotheses>", str(messages))
                    inserted = messages[1]["content"].split("<hypothesis>\n", 1)[1].split("\n</hypothesis>", 1)[0]
                    self.assertIn(inserted, hs.values())
            runner.diagnose(test)
            runner.analyze(test)
            summary = read_json(test / "results/question_analysis.json")
            self.assertEqual(summary["new_judgment_records"], 68)
            effect = next(row for row in summary["effects"] if row["condition"] == "extractive_general")
            self.assertEqual(effect["compression_failed_documents_q0"], 1)
            self.assertEqual(effect["compression_failed_documents_q17"], 0)
            self.assertEqual(effect["questions"], 17)
            self.assertGreater(effect["accuracy_gain"], 0)
            self.assertEqual(len(read_jsonl(test / "predictions.jsonl")), 85)
            self.assertEqual(before, {name: file_hash(main / name) for name in before})
            self.assertEqual(experiment.compression_messages, main_messages)
            with self.assertRaisesRegex(ValueError, "already initialized"):
                runner.initialize_test(main, check, root / "another_test")
            with self.assertRaisesRegex(ValueError, "Dev is technical"):
                runner.analyze(check)
            saved = read_json(test / "prompts.json")
            write_json(test / "prompts.json", {**saved, "tampered": True})
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                load_question_run(test)


if __name__ == "__main__":
    unittest.main()
