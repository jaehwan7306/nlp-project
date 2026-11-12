"""Synthetic, CPU-only checks. Never download data/tokenizers or invoke a model."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contract_experiment import CONDITIONS, LABELS
from contract_experiment.data import (audit_data, build_units, exclusion_decisions, prepare_data, load_prepared)
from contract_experiment.experiment import (compress, diagnose, freeze, initialize, judge, validated_results)
from contract_experiment.metrics import analyze, cluster_bootstrap, confusion_counts, scores
from contract_experiment.prompts import (EXTRACTIVE_TASK, PRIORITY, compression_messages, parse_label, select_units)
from contract_experiment.settings import ROOT, load_config
from contract_experiment.runtime import first_stop
from contract_experiment.review import write_review_report
from contract_experiment.storage import (append_jsonl, read_csv, read_json, read_jsonl,
                                         unique_index, write_json, digest, file_hash, write_csv)


def document(key, text, labels=None):
    split, doc_id = key.split(":")
    annotations = {f"nda-{i + 1}": {"choice": (labels or LABELS)[i % len(labels or LABELS)], "spans": []}
                   for i in range(17)}
    for annotation in annotations.values():
        if annotation["choice"] != "NotMentioned":
            annotation["spans"] = [0]
    spans = [[0, len(text)]]
    return {"key": key, "id": int(doc_id), "split": split, "text": text, "spans": spans,
            "units": build_units(text, spans), "annotations": annotations}


class CoreTests(unittest.TestCase):
    def test_priority_is_the_only_within_mode_difference_and_no_question_or_length(self):
        units = build_units("A must act. B may refuse.", [[0, 11], [12, 25]])
        for mode in ("generative", "extractive"):
            general = compression_messages(mode + "_general", units)
            priority = compression_messages(mode + "_priority", units)
            self.assertEqual(general[0], priority[0])
            self.assertEqual(general[1]["content"], priority[1]["content"].replace("\n" + PRIORITY, ""))
            lower = " ".join(item["content"] for item in general).lower()
            for forbidden in ("target", "percent", "token", "budget", "hypothesis", "nda-"):
                self.assertNotIn(forbidden, lower)

    def test_source_ranges_gaps_and_whole_unit_selection(self):
        text = "HEADING\nA must act.\nUnless excused.\nTAIL"
        a, b = text.index("A must"), text.index("Unless")
        spans = [[a, text.index("\n", a)], [b, text.index("\n", b)]]
        units = build_units(text, spans)
        self.assertEqual([u["kind"] for u in units], ["gap", "official", "official", "gap"])
        for unit in units:
            self.assertEqual(unit["text"], text[unit["start"]:unit["end"]])
        result = select_units("[3, 1, 3]", units)
        self.assertEqual(result["selected_ids"], [1, 3])
        self.assertEqual(result["text"], "HEADING\n\n\nUnless excused.")
        self.assertTrue(result["duplicate_ids"])
        self.assertEqual(select_units("[]", units)["text"], "")
        self.assertEqual(select_units("```json\n[2]\n```", units)["selected_ids"], [2])
        for output in ("[999]", "[true]", '["1"]', "Here: [1]", '{"ids":[1]}', "[1,"):
            self.assertEqual(select_units(output, units)["status"], "compression_failed")
        with self.assertRaises(ValueError):
            build_units(text, [[0, 10], [8, 12]])

    def test_label_parsing_and_failures_count_as_incorrect(self):
        for text, expected in (("entailment.", "Entailment"), ("not mentioned", "NotMentioned"),
                               ('{"label":"Contradiction"}', "Contradiction"),
                               ("Entailment or Contradiction", None), ("No label", None),
                               ("nonEntailment", None)):
            self.assertEqual(parse_label(text), expected)
        result = scores(confusion_counts([{"gold": "Entailment", "prediction": "Entailment"},
                                          {"gold": "Entailment", "prediction": None}]))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3) / 3)

    def test_dialogue_stop_at_first_or_later_line(self):
        stops = ["\nHuman:", "\nUser:"]
        self.assertEqual(first_stop("Human: fabricated dialogue", stops), (0, "\nHuman:"))
        self.assertEqual(first_stop("Summary.\nUser: follow-up", stops), (8, "\nUser:"))
        self.assertIsNone(first_stop("The term Human: appears within this source sentence.", stops))

    def test_explicit_human_review_actions_and_test_preference(self):
        docs = [document("test:1", "same", ["Entailment"]), document("test:2", "same", ["Entailment"]),
                document("dev:3", "same", ["Entailment"]), document("test:4", "other", ["Entailment"]),
                document("test:5", "other", ["Contradiction"])]
        links = [("test:1", "test:2"), ("test:2", "dev:3"), ("test:4", "test:5")]
        pairs = [{"pair_id": f"{a}|{b}", "left": a, "right": b, "review_required": True} for a, b in links]
        reviews = [{"pair_id": p["pair_id"], "decision": "duplicate", "notes": "human confirmed same content"} for p in pairs]
        reviews[2]["decision"] = "exclude_both"
        reviews[2]["notes"] = "human reviewed the labels and chose to exclude both"
        excluded = exclusion_decisions(docs, pairs, reviews)
        self.assertEqual({r["document"] for r in excluded}, {"test:2", "dev:3", "test:4", "test:5"})
        self.assertEqual(next(r["kept_document"] for r in excluded if r["document"] == "dev:3"), "test:1")
        self.assertEqual({r["decision"] for r in excluded if r["document"] in {"test:4", "test:5"}}, {"exclude_both"})
        self.assertTrue(all(r["review_notes"] for r in excluded))
        # Different Gold labels do not change a human's explicit keep-one decision.
        reviews[2]["decision"] = "duplicate"
        keep_one = exclusion_decisions(docs, pairs, reviews)
        self.assertNotIn("test:4", {r["document"] for r in keep_one})
        self.assertIn("test:5", {r["document"] for r in keep_one})
        # Exclude-both is a human action even when both Gold labels agree.
        docs[4]["annotations"] = {key: dict(value) for key, value in docs[3]["annotations"].items()}
        reviews[2]["decision"] = "exclude_both"
        remove_both = exclusion_decisions(docs, pairs, reviews)
        self.assertTrue({"test:4", "test:5"}.issubset({r["document"] for r in remove_both}))
        reviews[2]["decision"] = "distinct"
        no_action = exclusion_decisions(docs, pairs, reviews)
        self.assertTrue({"test:4", "test:5"}.isdisjoint({r["document"] for r in no_action}))
        reviews[0]["notes"] = ""
        with self.assertRaises(ValueError):
            exclusion_decisions(docs, pairs, reviews)

    def test_conflicting_overlapping_review_actions_stop_instead_of_guessing(self):
        docs = [document(f"test:{i}", "same", ["Entailment"]) for i in range(1, 4)]
        pairs = [{"pair_id": "test:1|test:2", "left": "test:1", "right": "test:2", "review_required": True},
                 {"pair_id": "test:1|test:3", "left": "test:1", "right": "test:3", "review_required": True}]
        reviews = [{"pair_id": pairs[0]["pair_id"], "decision": "duplicate", "notes": "keep one"},
                   {"pair_id": pairs[1]["pair_id"], "decision": "exclude_both", "notes": "remove both"}]
        with self.assertRaisesRegex(ValueError, "Conflicting reviews"):
            exclusion_decisions(docs, pairs, reviews)

    def test_journal_resume_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            append_jsonl(path, {"id": 1})
            with path.open("ab") as handle:
                handle.write(b'{"id":')
            with self.assertRaises(ValueError):
                read_jsonl(path)
            self.assertEqual(read_jsonl(path, repair=True), [{"id": 1}])
            self.assertEqual(len(list(Path(folder).glob("*.partial-*"))), 1)
            append_jsonl(path, {"id": 2})
            self.assertEqual(len(read_jsonl(path)), 2)
            with self.assertRaises(ValueError):
                unique_index([{"id": 1}, {"id": 1}], ("id",))

    def test_paired_document_bootstrap(self):
        rows = []
        for doc, correct in (("test:1", True), ("test:2", False)):
            for hid in range(17):
                rows.append({"document": doc, "condition": "original", "hypothesis": hid,
                             "gold": "Entailment", "prediction": "Entailment"})
                rows.append({"document": doc, "condition": "generative_general", "hypothesis": hid,
                             "gold": "Entailment", "prediction": "Entailment" if correct else None})
        result = cluster_bootstrap(rows, ("original", "generative_general"), repetitions=100, seed=42)
        self.assertEqual(result["generative_general"]["accuracy_drop"], [0.0, 1.0])
        self.assertEqual(result["original"]["accuracy_drop"], [0.0, 0.0])

    def test_config_rejects_length_target(self):
        config = load_config(ROOT / "config.toml")
        config["generation"]["target_retention"] = 0.3
        from contract_experiment.settings import validate_config
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_review_report_shows_differences_and_preserves_inputs(self):
        left = document("test:10", "Disclosure is prohibited.\nException: court order.\n<script>alert(1)</script>", ["Entailment"])
        right = document("test:20", "Disclosure is prohibited.\nException: law enforcement.\n<script>alert(1)</script>", ["Entailment"])
        right["annotations"]["nda-1"]["choice"] = "Contradiction"
        pair = {"pair_id": "test:10|test:20", "left": left["key"], "right": right["key"],
                "kind": "similar", "similarity": 0.96, "label_conflicts": ["nda-1"],
                "evidence_text_equal": False, "review_required": True}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            audit = {"pairs": [pair]}
            audit["audit_digest"] = digest(audit)
            write_json(root / "audit.json", audit)
            write_json(root / "candidate_documents.json", {"documents": [left, right],
                "hypotheses": {f"nda-{i + 1}": f"Claim {i + 1}" for i in range(17)}})
            write_csv(root / "review.csv", [{"pair_id": pair["pair_id"], "decision": "", "notes": ""}],
                      ["pair_id", "decision", "notes"])
            protected = {name: file_hash(root / name) for name in ("audit.json", "candidate_documents.json", "review.csv")}
            path = write_review_report(root / "audit.json")
            content = path.read_text(encoding="utf-8")
            self.assertIn('<table class="diff"', content)
            self.assertIn('class="conflict"', content)
            self.assertIn("Contradiction", content)
            self.assertIn("court order", content)
            self.assertIn("law enforcement", content)
            self.assertIn("chars [0,", content)
            self.assertNotIn("<script>alert(1)</script>", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertEqual(protected, {name: file_hash(root / name) for name in protected})
            self.assertIn("처리 방식 미선택", content)
            write_csv(root / "review.csv", [{"pair_id": pair["pair_id"], "decision": "duplicate", "notes": "keep one"}],
                      ["pair_id", "decision", "notes"])
            keep_one = write_review_report(root / "audit.json").read_text(encoding="utf-8")
            self.assertIn("test:10 유지 / test:20 제외", keep_one)
            write_csv(root / "review.csv", [{"pair_id": pair["pair_id"], "decision": "exclude_both", "notes": "remove both"}],
                      ["pair_id", "decision", "notes"])
            remove_both = write_review_report(root / "audit.json").read_text(encoding="utf-8")
            self.assertIn("test:10 및 test:20 모두 제외", remove_both)
            with self.assertRaises(ValueError):
                write_review_report(root / "audit.json", root / "review.csv")


class FakeTokenizer:
    chat_template = "synthetic-template"

    def encode(self, text, **kwargs):
        return text.split()

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)


class FakeModel:
    """Deterministic stub; no torch/transformers import or network calls."""
    calls = []

    def __init__(self, config, lock, role):
        self.role = role
        self.details = {"role": role, "load_seconds": 0.0, "fixture_only": True}

    def generate(self, messages):
        self.calls.append((self.role, messages))
        if self.role == "compressor":
            if EXTRACTIVE_TASK in messages[1]["content"]:
                text = "[9999]" if PRIORITY in messages[1]["content"] else "[1, 1]"
            else:
                text = "A faithful synthetic summary."
        else:
            text = "Entailment"
        return {"text": text, "raw_text": text, "raw_token_ids": [], "stop_reason": "eos",
                "cap_reached": False, "dialogue_flag": False, "input_tokens": 10,
                "generated_tokens": 2, "generation_seconds": 0.0, "rendered_prompt_sha256": "fixture"}

    def close(self):
        pass


class WorkflowTests(unittest.TestCase):
    def test_synthetic_main_workflow_freeze_resume_and_failure_denominator(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            data, audit, prepared = root / "data", root / "audit", root / "prepared"
            hypotheses = {f"nda-{i + 1}": {"hypothesis": f"Fixture hypothesis {i + 1}."} for i in range(17)}
            for split, doc_id, text in (("train", 1, "ALPHA source contract."), ("dev", 2, "BETA development contract."),
                                        ("test", 3, "GAMMA held out contract.")):
                doc = document(f"{split}:{doc_id}", text)
                write_json(data / f"{split}.json", {"labels": hypotheses, "documents": [{
                    "id": doc_id, "text": text, "spans": doc["spans"],
                    "annotation_sets": [{"annotations": doc["annotations"]}]}]})
            with patch("contract_experiment.data.EXPECTED_COUNTS", dict.fromkeys(("train", "dev", "test"), 1)):
                audit_data(data, audit)
                prepare_data(audit / "audit.json", audit / "review.csv", prepared)
            config_path = root / "config.toml"
            config_path.write_text((ROOT / "config.toml").read_text(encoding="utf-8").replace("bootstrap_repetitions = 10000", "bootstrap_repetitions = 30"), encoding="utf-8")
            models = root / "models.json"
            config = load_config(config_path)
            write_json(models, {"models": {role: {"repo_id": name, "revision": "a" * 40} for role, name in config["models"].items()}})
            dev, test, protocol = root / "dev", root / "test", root / "protocol"
            FakeModel.calls = []
            with patch("contract_experiment.experiment.load_tokenizer", return_value=FakeTokenizer()), patch("contract_experiment.experiment.LocalModel", FakeModel):
                initialize(config_path, prepared, models, "dev", dev)
                judge(dev, ["original"])
                compress(dev)
                judge(dev, resume=True)
                diagnose(dev)
                freeze(dev, protocol)
                initialize(config_path, prepared, models, "test", test, protocol=protocol)
                with self.assertRaises(ValueError):
                    initialize(config_path, prepared, models, "test", root / "test-again", protocol=protocol)
                judge(test, ["original"])
                analyze(test, ["original"])
                self.assertTrue((test / "results" / "baseline" / "metrics.csv").exists())
                compress(test)
                judge(test, resume=True)
                before = len(FakeModel.calls)
                compress(test, resume=True)
                judge(test, resume=True)
                self.assertEqual(len(FakeModel.calls), before)
                diagnose(test)
                analyze(test)
            rows = read_jsonl(test / "predictions.jsonl")
            self.assertEqual(len(rows), 5 * 17)
            summary = read_json(test / "results" / "analysis.json")
            failed = next(row for row in summary["metrics"] if row["condition"] == "extractive_priority")
            self.assertEqual(failed["accuracy"], 0.0)
            self.assertEqual(failed["questions"], 17)
            self.assertEqual(failed["compression_failed_documents"], 1)
            self.assertEqual(failed["length_available_documents"], 0)
            self.assertIsNone(failed["retention_median"])
            self.assertFalse(read_json(test / "diagnostics" / "technical_summary.json")["accuracy_computed"])
            for role, messages in FakeModel.calls:
                if role == "compressor":
                    self.assertNotIn("Fixture hypothesis", str(messages))
                    self.assertNotIn("annotations", str(messages))
            dataset = read_json(test / "dataset.json")
            validated_results(test, dataset)
            (test / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validated_results(test, dataset)
            legacy = read_json(prepared / "manifest.json")
            legacy.pop("review_action_policy")
            write_json(prepared / "manifest.json", legacy)
            with self.assertRaisesRegex(ValueError, "previous automatic review policy"):
                load_prepared(prepared)


if __name__ == "__main__":
    unittest.main()
