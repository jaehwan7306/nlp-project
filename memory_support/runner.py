"""A recorded runtime extension; never rewrite a frozen source/protocol hash."""

import argparse
import shutil
import sys
from pathlib import Path

from contract_experiment import CONDITIONS
from contract_experiment.prompts import compression_messages, judge_messages
from contract_experiment.settings import ROOT, environment, load_run
from contract_experiment.storage import (append_jsonl, digest, file_hash, now,
    read_json, read_jsonl, run_lock, unique_index, write_json)
from .attention import memory_model, policy_spec


def comparable(details):
    return {key: value for key, value in details.items() if key != "load_seconds"}


def check_dev(run_path, output):
    from contract_experiment.runtime import LocalModel

    run, manifest, config, models, dataset = load_run(run_path)
    if manifest["split"] != "dev" or read_json(run / "environment.json") != environment():
        raise ValueError("Check requires a completed greedy Dev in the same environment.")
    contexts = unique_index(read_jsonl(run / "contexts.jsonl"), ("document", "condition"))
    predictions = unique_index(read_jsonl(run / "predictions.jsonl"), ("document", "condition", "hypothesis"))
    if len(contexts) != manifest["expected_contexts"] or len(predictions) != manifest["expected_judgments"]:
        raise ValueError("The source Dev must be complete.")
    originals = [row for row in contexts.values() if row["condition"] == "original"]
    longest = min(originals, key=lambda row: (-row["context_tokens"], row["document"]))["document"]
    doc = next(doc for doc in dataset["documents"] if doc["key"] == longest)
    hypothesis = next(iter(dataset["hypotheses"]))
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Check directory contains files. Preserve it and choose a new --out.")
    out.mkdir(parents=True, exist_ok=True)
    spec = policy_spec()
    report = {"created_utc": now(), "status": "in_progress", "passed": False,
        "policy": spec, "source_run": str(run.resolve()), "identity": manifest["identity"],
        "environment": environment(), "contexts_sha256": file_hash(run / "contexts.jsonl"),
        "predictions_sha256": file_hash(run / "predictions.jsonl"),
        "selection": "longest Original Dev document; all four compression tasks and all five judgments for the first fixed hypothesis",
        "document": longest, "hypothesis": hypothesis, "checks": [], "runtime": {}}
    write_json(out / "report.json", report)
    model_type = memory_model(LocalModel, spec)
    for role in ("judge", "compressor"):
        model = model_type(config, models, role)
        try:
            actual = comparable(model.details)
            expected = comparable(read_json(run / f"runtime_{role}.json"))
            if actual != expected or actual.get("decoding_policy") != "explicit_greedy_v1":
                raise ValueError(f"Base {role} runtime differs from the completed greedy Dev.")
            report["runtime"][role] = actual
            conditions = CONDITIONS if role == "judge" else CONDITIONS[1:]
            for condition in conditions:
                print(f"[memory-check {len(report['checks']) + 1}/9] {longest} {role} {condition}", flush=True)
                if role == "judge":
                    stored = predictions[(longest, condition, hypothesis)]["generation"]
                    messages = judge_messages(contexts[(longest, condition)]["text"], dataset["hypotheses"][hypothesis])
                else:
                    stored = contexts[(longest, condition)]["generation"]
                    messages = compression_messages(condition, doc["units"])
                observed = model.generate(messages)
                equal = {field: observed[field] == stored[field] for field in
                    ("raw_token_ids", "text", "stop_reason", "rendered_prompt_sha256", "input_tokens")}
                report["checks"].append({"role": role, "condition": condition, "equal": equal,
                    "passed": all(equal.values()), "stored_tokens_sha256": digest(stored["raw_token_ids"]),
                    "observed_tokens_sha256": digest(observed["raw_token_ids"]),
                    "attention_memory_policy": observed["attention_memory_policy"]})
                write_json(out / "report.json", report)
        finally:
            model.close()
    if (file_hash(run / "contexts.jsonl") != report["contexts_sha256"] or
            file_hash(run / "predictions.jsonl") != report["predictions_sha256"]):
        raise ValueError("Source Dev changed during verification.")
    report["status"] = "complete"
    report["passed"] = all(check["passed"] for check in report["checks"])
    write_json(out / "report.json", report)
    print(f"Dev memory check: {'PASS' if report['passed'] else 'FAIL'}; report: {out / 'report.json'}")
    return 0 if report["passed"] else 2


def verify_report(report, manifest, run):
    expected = {("judge", condition) for condition in CONDITIONS} | {
                ("compressor", condition) for condition in CONDITIONS[1:]}
    if (report.get("status") != "complete" or report.get("passed") is not True or
            len(report.get("checks", [])) != 9 or
            {(check["role"], check["condition"]) for check in report.get("checks", [])} != expected or
            not all(check["passed"] and all(check["equal"].values()) for check in report["checks"])):
        raise ValueError("A passing 9-output Dev memory check is required. See README.")
    if (report["policy"] != policy_spec() or report["identity"] != manifest["identity"] or
            report["environment"] != environment()):
        raise ValueError("Memory check code, base identity or environment changed.")
    base_runtime = read_json(run / "frozen_protocol.json")["runtime"]
    if report["runtime"] != base_runtime:
        raise ValueError("Memory check base runtime differs from the frozen protocol.")


def verify_new_rows(run, record, repair=False):
    expected = digest(record["policy"])
    for filename, boundary in record["journal_boundaries"].items():
        rows = read_jsonl(run / filename, repair=repair)
        before = record["journal_prefix_digests"][filename]
        if len(rows) < boundary or digest(rows[:boundary]) != before:
            raise ValueError(f"Pre-extension journal records changed: {filename}")
        for row in rows[boundary:]:
            generation = row.get("generation")
            if generation is not None and generation.get("attention_memory_policy", {}).get("policy_sha256") != expected:
                raise ValueError("Post-extension output lacks the recorded memory policy. Use run_memory_safe.py for model stages.")


def adopt(run, manifest, report, resume=False):
    path = run / "runtime_memory_policy.json"
    if path.exists():
        record = read_json(path)
        if record["policy"] != policy_spec() or record["check_report_sha256"] != digest(report):
            raise ValueError("Recorded memory policy/check changed; do not mix execution policies.")
        for name, sha in record["policy"]["sources"].items():
            if file_hash(run / "memory_extension_source" / name) != sha:
                raise ValueError("Archived memory extension source changed.")
        verify_new_rows(run, record, repair=resume)
        return record
    journals = {name: read_jsonl(run / name, repair=resume) for name in ("contexts.jsonl", "predictions.jsonl")}
    record = {"created_utc": now(), "base_identity": manifest["identity"], "policy": policy_spec(),
        "check_report_sha256": digest(report), "check_report": report,
        "journal_boundaries": {name: len(rows) for name, rows in journals.items()},
        "journal_prefix_digests": {name: digest(rows) for name, rows in journals.items()},
        "note": "Recorded implementation extension after OOM; earlier records preserved. Numerical identity is checked on a Dev sample, not guaranteed for all inputs."}
    for name in record["policy"]["sources"]:
        target = run / "memory_extension_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(path, record)
    append_jsonl(run / "events.jsonl", {"event": "memory_policy_adopted", "time": now(),
        "policy_sha256": digest(record["policy"]), "preserved_rows": record["journal_boundaries"]})
    return record


def add_report_note(run):
    record = read_json(run / "runtime_memory_policy.json")
    for analysis_path in (run / "results" / "analysis.json", run / "results" / "baseline" / "analysis.json"):
        if analysis_path.exists():
            analysis = read_json(analysis_path)
            analysis["attention_memory_execution"] = {
                "policy_id": record["policy"]["id"], "policy_sha256": digest(record["policy"]),
                "preserved_journal_prefix": record["journal_boundaries"],
                "provenance_file": "runtime_memory_policy.json",
                "numerical_check_scope": "9 Dev outputs on the longest Original Dev document",
                "timing_caveat": "Includes an implementation change; not a uniform-kernel benchmark"}
            write_json(analysis_path, analysis)
    for report_path in (run / "results" / "report.md", run / "results" / "baseline" / "report.md"):
        if report_path.exists():
            text = report_path.read_text(encoding="utf-8")
            marker = "## Attention memory execution record"
            if marker not in text:
                text += ("\n\n" + marker + "\n\nThis run adopted a separately archived headwise math attention extension after an OOM. "
                    "Earlier judgments were retained following a passing Dev output check. Full inputs and model settings were preserved; "
                    "universal bitwise equivalence is not guaranteed. Read runtime_memory_policy.json for the boundary, source hashes, and verification. "
                    "Timing includes the implementation change and is not a uniform-kernel benchmark.\n")
                report_path.write_text(text, encoding="utf-8")


def execute(arguments, report_path):
    from contract_experiment import cli, experiment

    args = cli.parser().parse_args(arguments)
    if args.command not in {"judge", "compress", "diagnose", "analyze"}:
        raise ValueError("Use the memory runner only for judge/compress/diagnose/analyze; use the normal entry point to initialize/freeze.")
    run, manifest, config, models, dataset = load_run(args.run)
    if manifest["split"] != "test":
        raise ValueError("This recovery runner is for Test; use check-dev for its Dev comparison.")
    report = read_json(report_path)
    verify_report(report, manifest, run)
    with run_lock(run):
        adopt(run, manifest, report, resume=getattr(args, "resume", False))
    original = experiment.LocalModel
    experiment.LocalModel = memory_model(original, policy_spec())
    try:
        result = cli.main(arguments)
    finally:
        experiment.LocalModel = original
    with run_lock(run):
        verify_new_rows(run, read_json(run / "runtime_memory_policy.json"))
        if result == 0 and args.command == "analyze":
            add_report_note(run)
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if sys.version_info[:2] != (3, 11):
        print("ERROR: Use Python 3.11.", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description="Full-context attention memory recovery; original source/protocol stays frozen.")
    parser.add_argument("--check-report", default="runs/memory_check/report.json")
    options, remaining = parser.parse_known_args(arguments)
    try:
        check_arguments = [argument for argument in remaining if argument != "--debug"]
        if check_arguments and check_arguments[0] == "check-dev":
            check = argparse.ArgumentParser(description="Nine greedy output comparisons; never select by Gold or accuracy.")
            check.add_argument("--run", default="runs/dev_greedy")
            check.add_argument("--out", default="runs/memory_check")
            args = check.parse_args(check_arguments[1:])
            return check_dev(args.run, args.out)
        if not remaining:
            parser.print_help()
            print("Use check-dev --run runs/dev_greedy, or judge/compress/diagnose/analyze --run runs/test_greedy.")
            return 0
        return execute(remaining, options.check_report)
    except KeyboardInterrupt:
        print("Interrupted. Retain completed rows and resume this runner with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        if "--debug" in remaining:
            import traceback
            traceback.print_exc()
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
