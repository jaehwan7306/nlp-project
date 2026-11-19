"""Reuse the existing compression and judgment stages with Q8 prompts."""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

from contract_experiment import CONDITIONS, experiment
from contract_experiment.storage import read_json, run_lock
from memory_support.attention import memory_model

from .assignment import compression_messages
from .state import initialize, load_subset_run


@contextmanager
def execution(path, resume):
    with run_lock(path):
        loaded = load_subset_run(path, repair=resume)
    run, _, _, _, dataset = loaded
    protocol = read_json(run / "frozen_protocol.json")
    assignment = read_json(run / "assignment.json")["by_document"]
    marker = protocol["question_context_policy"]
    base_model, base_load, base_messages = experiment.LocalModel, experiment.load_run, experiment.compression_messages
    parent = memory_model(base_model, protocol["memory_policy"], stamp_runtime=True)

    class SubsetModel(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # model_details checks these against the frozen runtime before generation.
            self.details = {**self.details, "question_context_policy": marker}

        def generate(self, messages):
            generated = super().generate(messages)
            generated["question_context_policy"] = marker
            return generated

    # The core compressor passes each document's units list unchanged. Use the
    # verified snapshot's list identity so identical text cannot mix assignments.
    shown_by_units = {id(doc["units"]): assignment[doc["key"]] for doc in dataset["documents"]}

    def load_verified(candidate):
        if Path(candidate).resolve() != run:
            raise ValueError("Q8 execution cannot switch run directories.")
        return loaded

    def messages(condition, units):
        return compression_messages(condition, units, dataset["hypotheses"], shown_by_units[id(units)])

    experiment.LocalModel, experiment.load_run, experiment.compression_messages = SubsetModel, load_verified, messages
    try:
        yield
    finally:
        experiment.LocalModel, experiment.load_run, experiment.compression_messages = base_model, base_load, base_messages


def parser():
    cli = argparse.ArgumentParser(description="Q8: balanced hypothesis subsets and paired comparison with Q0/Q17.")
    cli.add_argument("--debug", action="store_true", help="show a traceback on error")
    stages = cli.add_subparsers(dest="command", required=True)
    init = stages.add_parser("init", help="verify Q0 and Q17, then save the deterministic Q8 assignments")
    init.add_argument("--main-run", default="runs/test_memory_safe")
    init.add_argument("--q17-run", default="runs/questions17")
    init.add_argument("--out", default="runs/questions8")
    for name in ("compress", "judge", "diagnose", "analyze"):
        stage = stages.add_parser(name)
        stage.add_argument("--run", default="runs/questions8")
        if name in {"compress", "judge"}:
            stage.add_argument("--resume", action="store_true")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info[:2] != (3, 11):
        print("ERROR: Use Python 3.11.", file=sys.stderr)
        return 2
    try:
        if args.command == "init":
            initialize(args.main_run, args.q17_run, args.out)
        elif args.command == "analyze":
            from .comparison import compare
            compare(args.run)
        else:
            resume = getattr(args, "resume", False)
            with execution(args.run, resume):
                match args.command:
                    case "compress":
                        experiment.compress(args.run, resume=resume)
                    case "judge":
                        experiment.judge(args.run, conditions=CONDITIONS[1:], resume=resume)
                    case "diagnose":
                        experiment.diagnose(args.run)
    except KeyboardInterrupt:
        print("Interrupted. Keep the Q8 run and resume the interrupted stage with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"ERROR: {exc}\nUse --debug before the command for a traceback.", file=sys.stderr)
        return 1
    return 0
