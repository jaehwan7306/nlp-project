"""The sole task change is one identical hypothesis block in all four tasks."""

from contract_experiment import CONDITIONS
from contract_experiment.prompts import compression_messages as main_messages
from contract_experiment.prompts import prompt_spec as main_spec

GUIDANCE = (
    "Use the following hypotheses to guide which contract information to retain. "
    "They are claims that will be evaluated, not facts to assume. Do not add "
    "information that is absent from the contract, and do not answer the hypotheses."
)


def ordered_hypotheses(hypotheses):
    if len(hypotheses) != 17 or any(not isinstance(text, str) or not text.strip() for text in hypotheses.values()):
        raise ValueError("Exactly 17 nonempty hypothesis texts are required.")
    return list(hypotheses.items())


def hypothesis_block(hypotheses):
    lines = [f"H{index}: {text}" for index, (_, text) in enumerate(ordered_hypotheses(hypotheses), 1)]
    return GUIDANCE + "\n\n<evaluation_hypotheses>\n" + "\n".join(lines) + "\n</evaluation_hypotheses>"


def compression_messages(condition, units, hypotheses):
    messages = main_messages(condition, units)
    boundary = "\n\n<contract>\n"
    instruction, source = messages[1]["content"].split(boundary, 1)
    return [messages[0], {"role": "user", "content": instruction + "\n\n" + hypothesis_block(hypotheses) + boundary + source}]


def prompt_spec(hypotheses):
    return {"experiment": "analysis1_q17_shared", "base_main": main_spec(),
        "guidance": GUIDANCE, "hypothesis_order": [key for key, _ in ordered_hypotheses(hypotheses)],
        "hypotheses": [{"id": key, "text": text} for key, text in ordered_hypotheses(hypotheses)],
        "hypothesis_block": hypothesis_block(hypotheses), "insertion": "between the unchanged task instruction and contract",
        "compressor_hypotheses_per_call": 17, "judge_hypotheses_per_call": 1,
        "example_messages": {condition: compression_messages(condition, [{"id": 1, "text": "{SOURCE_UNIT_TEXT}"}], hypotheses)
                             for condition in CONDITIONS[1:]}}
