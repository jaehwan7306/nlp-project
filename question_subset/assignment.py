"""Assign hypotheses by ID and filter the existing Q17 prompt."""

import random
from collections import Counter

from contract_experiment import CONDITIONS
from question_support.prompts import (GUIDANCE, compression_messages as q17_messages,
                                     hypothesis_block, ordered_hypotheses)


EXPERIMENT = "analysis1_q8_balanced"
PER_DOCUMENT = 8
SEED = 42


def balanced_assignment(document_keys, hypothesis_ids):
    documents, ids = sorted(document_keys), list(hypothesis_ids)
    if not documents or len(set(documents)) != len(documents) or len(ids) != 17 or len(set(ids)) != 17:
        raise ValueError("Q8 needs unique document keys and exactly 17 hypothesis IDs.")
    rng = random.Random(SEED)
    rng.shuffle(documents)
    selected = {}
    for start in range(0, len(documents), len(ids)):
        order = rng.sample(ids, k=len(ids))
        cycle = order + order
        for offset, doc in enumerate(documents[start:start + len(ids)]):
            # Deal eight consecutive positions per document from a shuffled cycle.
            # A full block gives every ID eight exposures; a partial block differs by at most one.
            first = (offset * PER_DOCUMENT) % len(ids)
            shown = set(cycle[first:first + PER_DOCUMENT])
            selected[doc] = [hid for hid in ids if hid in shown]
    counts = Counter(hid for group in selected.values() for hid in group)
    return {"seed": SEED, "algorithm": "shuffled_document_blocks_and_hypothesis_cycles_v1",
            "hypothesis_order": ids, "shown_per_document": PER_DOCUMENT,
            "by_document": dict(sorted(selected.items())),
            "exposure_counts": {hid: {"shown": counts[hid], "omitted": len(documents) - counts[hid]} for hid in ids}}


def compression_messages(condition, units, hypotheses, selected_ids):
    shown = set(selected_ids)
    if len(selected_ids) != PER_DOCUMENT or len(shown) != PER_DOCUMENT or not shown <= hypotheses.keys():
        raise ValueError("Each Q8 prompt must show eight different official hypotheses.")
    messages = q17_messages(condition, units, hypotheses)
    lines = [f"H{number}: {text}" for number, (hid, text) in enumerate(ordered_hypotheses(hypotheses), 1)
             if hid in shown]
    block = GUIDANCE + "\n\n<evaluation_hypotheses>\n" + "\n".join(lines) + "\n</evaluation_hypotheses>"
    messages[1]["content"] = messages[1]["content"].replace(hypothesis_block(hypotheses), block, 1)
    return messages


def prompt_spec(dataset, assignment, q17_prompts):
    example_key = next(iter(assignment["by_document"]))
    example_units = [{"id": 1, "text": "{SOURCE_UNIT_TEXT}"}]
    return {"experiment": EXPERIMENT, "main_prompts": q17_prompts["base_main"], "guidance": GUIDANCE,
            "official_hypotheses": q17_prompts["hypotheses"], "assignment_file": "assignment.json",
            "numbering": "retain Q17 H numbers and relative order; remove omitted hypothesis lines",
            "insert_after_task_before_contract": True, "judge_hypotheses_per_call": 1,
            "example_document": example_key,
            "example_messages": {condition: compression_messages(condition, example_units, dataset["hypotheses"],
                                                                 assignment["by_document"][example_key])
                                 for condition in CONDITIONS[1:]}}
