"""Explicit additional-analysis stages; Main files are always read-only."""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from contract_experiment import CONDITIONS
from contract_experiment import experiment, metrics
from contract_experiment.settings import environment
from contract_experiment.storage import digest, file_hash, now, read_json, read_jsonl, run_lock, write_json
from memory_support.attention import memory_model
from .prompts import compression_messages, prompt_spec
from .state import (EXPERIMENT, initialize, inspect_main, load_question_run,
                    preflight_documents, question_protocol)


@contextmanager
def execution(path, resume=False):
    run = Path(path)
    with run_lock(run):
        loaded = load_question_run(run, repair=resume)
    _, manifest, _, _, dataset = loaded
    protocol = read_json(run / "frozen_protocol.json")
    original_model = experiment.LocalModel
    original_messages = experiment.compression_messages
    original_load = experiment.load_run
    original_metrics_load = metrics.load_run
    parent = memory_model(original_model, protocol["memory_policy"], stamp_runtime=True)

    class QuestionModel(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            actual = {k: v for k, v in self.details.items() if k != "load_seconds"}
            expected = {k: v for k, v in protocol["runtime"][self.role].items() if k != "question_context_policy"}
            if actual != expected:
                self.close()
                raise ValueError("Question model settings differ from Main. No generation was started.")
            self.details = {**self.details, "question_context_policy": protocol["question_context_policy"]}
            print(f"Question policy: compressor=17 together / judge=1 at a time / role={self.role}", flush=True)

        def generate(self, messages):
            generated = super().generate(messages)
            generated["question_context_policy"] = protocol["question_context_policy"]
            return generated

    experiment.LocalModel = QuestionModel
    experiment.compression_messages = lambda condition, units: compression_messages(condition, units, dataset["hypotheses"])
    experiment.load_run = load_question_run
    metrics.load_run = load_question_run
    try:
        yield loaded
    finally:
        experiment.LocalModel = original_model
        experiment.compression_messages = original_messages
        experiment.load_run = original_load
        metrics.load_run = original_metrics_load


def compress(path, resume=False):
    with execution(path, resume):
        experiment.compress(path, resume)
    load_question_run(path)


def judge(path, resume=False):
    with execution(path, resume) as (_, manifest, *_):
        # Test Original is an exact reference copy; it is never generated again.
        conditions = CONDITIONS[1:] if manifest["split"] == "test" else CONDITIONS
        experiment.judge(path, conditions, resume)
    load_question_run(path)


def diagnose(path):
    with execution(path):
        experiment.diagnose(path)
    run, manifest, *_ = load_question_run(path)
    summary_path = run / "diagnostics/technical_summary.json"
    summary = read_json(summary_path)
    rows = read_jsonl(run / "predictions.jsonl")
    summary["question_analysis"] = {"experiment": EXPERIMENT, "hypotheses_per_compression": 17,
        "hypotheses_per_judgment": 1, "reused_original_judgments": manifest["reused_original_judgments"],
        "new_judgment_records": sum(r["condition"] != "original" or manifest["split"] != "test" for r in rows),
        "new_model_judgments": sum(r["generation"] is not None and (r["condition"] != "original" or manifest["split"] != "test") for r in rows)}
    write_json(summary_path, summary)
    return summary


def preflight(main_path, dev_path, output, documents=1, resume=False):
    main = inspect_main(main_path, require_environment=True)
    out = Path(output)
    if (out / "manifest.json").exists() and resume:
        _, manifest, *_ = load_question_run(out, repair=True)
        if manifest["split"] != "dev" or manifest["identity"] != question_protocol(main)["identity"]:
            raise ValueError("--resume preflight requires a Q17 Dev run.")
    else:
        selected, source = preflight_documents(main, dev_path, documents)
        initialize(main, selected, "dev", out, dev_source=source)
    compress(out, resume=resume)
    judge(out, resume=True)
    summary = diagnose(out)
    _, manifest, *_ = load_question_run(out)
    complete = summary["context_count"] == manifest["expected_contexts"] and summary["prediction_count"] == manifest["expected_judgments"]
    failures = sum(c["compression_failed"] + c["parsing_failures"] for c in summary["conditions"].values())
    receipt = {"completed_utc": now(), "experiment": EXPERIMENT, "identity": manifest["identity"],
        "passed": complete and failures == 0, "accuracy_computed": False,
        "context_count": summary["context_count"], "prediction_count": summary["prediction_count"],
        "contexts_sha256": summary["contexts_sha256"], "predictions_sha256": summary["predictions_sha256"],
        "technical_summary_sha256": file_hash(out / "diagnostics/technical_summary.json"),
        "acceptance": "complete execution and valid IDs/labels; no accuracy, output-length or content-quality selection"}
    write_json(out / "preflight.json", receipt)
    print(f"Q17 preflight: {'PASS' if receipt['passed'] else 'FAIL'}; no accuracy calculated. See {out / 'diagnostics'}")
    return 0 if receipt["passed"] else 2


def initialize_test(main_path, preflight_path, output):
    main = inspect_main(main_path, require_environment=True)
    check = Path(preflight_path)
    _, checked, *_ = load_question_run(check)
    receipt = read_json(check / "preflight.json")
    if (checked["split"] != "dev" or receipt.get("passed") is not True or receipt.get("accuracy_computed") is not False or
            receipt["identity"] != question_protocol(main)["identity"] or
            receipt["contexts_sha256"] != file_hash(check / "contexts.jsonl") or
            receipt["predictions_sha256"] != file_hash(check / "predictions.jsonl") or
            receipt["technical_summary_sha256"] != file_hash(check / "diagnostics/technical_summary.json")):
        raise ValueError("Complete a matching Q17 Dev preflight before initializing Test.")
    for role in ("compressor", "judge"):
        actual = {k: v for k, v in read_json(check / f"runtime_{role}.json").items() if k != "load_seconds"}
        if actual != question_protocol(main)["runtime"][role]:
            raise ValueError("Preflight model runtime does not match the Q17 protocol.")
    with run_lock(check):
        if (check / "test_run.json").exists():
            raise ValueError("This Q17 preflight already initialized Test. Resume that run.")
        initialize(main, main["dataset"]["documents"], "test", output, preflight_receipt=receipt)
        with (check / "test_run.json").open("x", encoding="utf-8") as handle:
            json.dump({"run": str(Path(output).resolve()), "started_utc": now()}, handle)


def analyze(path):
    run, manifest, *_ = load_question_run(path)
    if manifest["split"] != "test":
        raise ValueError("Dev is technical only. Analyze the Q17 Test, not its preflight.")
    with execution(path):
        metrics.analyze(path)
    from .comparison import compare
    compare(path)


def parser():
    cli = argparse.ArgumentParser(description="Additional analysis 1: Q17-aware compression, same frozen Main settings.")
    cli.add_argument("--debug", action="store_true")
    commands = cli.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("preflight", help="small Dev technical run; no accuracy")
    smoke.add_argument("--main-run", default="runs/test_memory_safe")
    smoke.add_argument("--dev-run", default="runs/dev_greedy")
    smoke.add_argument("--out", default="runs/questions17_preflight")
    smoke.add_argument("--documents", type=int, default=1)
    smoke.add_argument("--resume", action="store_true")
    init = commands.add_parser("init", help="freeze Q17 Test; reuse only the Main Original baseline")
    init.add_argument("--main-run", default="runs/test_memory_safe")
    init.add_argument("--preflight", default="runs/questions17_preflight")
    init.add_argument("--out", default="runs/questions17")
    for name in ("compress", "judge", "diagnose", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--run", default="runs/questions17")
        if name in ("compress", "judge"):
            command.add_argument("--resume", action="store_true")
    prompts = commands.add_parser("show-prompts", help="print/save exact instructions and all 17 hypotheses; no models")
    prompts.add_argument("--main-run", default="runs/test_memory_safe")
    prompts.add_argument("--out")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info[:2] != (3, 11):
        print("ERROR: Use Python 3.11.", file=sys.stderr)
        return 2
    try:
        match args.command:
            case "preflight":
                return preflight(args.main_run, args.dev_run, args.out, args.documents, args.resume)
            case "init":
                initialize_test(args.main_run, args.preflight, args.out)
            case "compress":
                compress(args.run, args.resume)
            case "judge":
                judge(args.run, args.resume)
            case "diagnose":
                diagnose(args.run)
            case "analyze":
                analyze(args.run)
            case "show-prompts":
                source = inspect_main(args.main_run)
                spec = prompt_spec(source["dataset"]["hypotheses"])
                if args.out:
                    write_json(args.out, spec)
                else:
                    print(json.dumps(spec, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("Interrupted. Keep the same run and resume its stage with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"ERROR: {exc}\nUse --debug before the command for a traceback.", file=sys.stderr)
        return 1
    return 0
