"""Check the assignment, prompt isolation and identical-item comparisons for Q8."""

import unittest

from contract_experiment import CONDITIONS
from question_subset.assignment import balanced_assignment, compression_messages
from question_subset.comparison import ARMS, bootstrap_weights, effect, intervals, pair_predictions
from question_support.prompts import compression_messages as q17_messages


HYPOTHESES = {f"h{i:02d}": f"Unique statement number {i:02d}." for i in range(1, 18)}


class QuestionSubsetTests(unittest.TestCase):
    def test_balanced_assignment_is_reproducible_and_preserves_order(self):
        documents = [f"test:{i}" for i in range(1, 120)]
        assignment = balanced_assignment(documents, HYPOTHESES)
        self.assertEqual(assignment, balanced_assignment(reversed(documents), HYPOTHESES))
        self.assertEqual(len(assignment["by_document"]), 119)
        self.assertEqual(list(assignment["exposure_counts"].values()), [{"shown": 56, "omitted": 63}] * 17)
        self.assertGreater(len({tuple(ids) for ids in assignment["by_document"].values()}), 17)
        for ids in assignment["by_document"].values():
            self.assertEqual(len(ids), 8)
            self.assertEqual(ids, [hid for hid in HYPOTHESES if hid in set(ids)])
        partial = balanced_assignment(documents[:-1], HYPOTHESES)
        self.assertEqual({row["shown"] for row in partial["exposure_counts"].values()}, {55, 56})

    def test_q8_only_removes_nine_q17_hypothesis_lines(self):
        documents = ["test:1", "test:2"]
        assignment = balanced_assignment(documents, HYPOTHESES)["by_document"]
        units = [{"id": 1, "text": "A contract that mentions H3: as literal text."}]
        for condition in CONDITIONS[1:]:
            q17 = q17_messages(condition, units, HYPOTHESES)
            for document in documents:
                selected = assignment[document]
                expected = q17[1]["content"]
                for number, (hid, text) in enumerate(HYPOTHESES.items(), 1):
                    if hid not in selected:
                        expected = expected.replace(f"H{number}: {text}\n", "", 1)
                self.assertEqual(compression_messages(condition, units, HYPOTHESES, selected),
                                 [q17[0], {"role": "user", "content": expected}])

    def test_exposure_mask_is_identical_across_arms_and_failures_stay_in_sample(self):
        documents = ["test:1", "test:2"]
        assignment = balanced_assignment(documents, HYPOTHESES)["by_document"]
        rows = {arm: [] for arm in ARMS}
        for doc in documents:
            for hid in HYPOTHESES:
                shown = hid in assignment[doc]
                for arm in ARMS:
                    failed = shown and arm == "q0"
                    rows[arm].append({"document": doc, "condition": "generative_general", "hypothesis": hid,
                                      "gold": "Entailment", "prediction": None if failed else "Entailment",
                                      "status": "parse_failed" if failed else "ok"})
        rows["q17"].reverse()
        pairs = pair_predictions(rows, assignment)
        shown = [row for row in pairs if row["shown_in_q8"]]
        omitted = [row for row in pairs if not row["shown_in_q8"]]
        self.assertEqual((len(shown), len(omitted)), (16, 18))
        self.assertEqual(effect(shown)["accuracy_q8_minus_q0"], 1.0)
        self.assertEqual(effect(omitted)["accuracy_q8_minus_q0"], 0.0)
        self.assertEqual(effect(shown)["accuracy_q8_minus_q17"], 0.0)
        ci = intervals(shown, documents, bootstrap_weights(documents, 20, 42))
        self.assertEqual(ci["accuracy_q8_minus_q0_ci_low"], 1.0)
        self.assertEqual(ci["accuracy_q8_minus_q0_ci_high"], 1.0)
        self.assertEqual(ci["macro_f1_q8_minus_q17_ci_high"], 0.0)
        rows["q17"].pop()
        with self.assertRaisesRegex(ValueError, "identical judgment keys"):
            pair_predictions(rows, assignment)


if __name__ == "__main__":
    unittest.main()
