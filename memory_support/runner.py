"""A recorded runtime extension; never rewrite a frozen source/protocol hash."""

import argparse
import shutil
import sys
from pathlib import Path

from contract_experiment import CONDITIONS
from contract_experiment.prompts import compression_messages, judge_messages, parse_label, select_units
from contract_experiment.settings import ROOT, environment, identity, load_config, load_run, validate_models
from contract_experiment.storage import (append_jsonl, digest, file_hash, now,
    read_json, read_jsonl, run_lock, unique_index, write_json)
from .attention import memory_model, policy_spec, runtime_policy_marker


CHECK_SCHEMA = "fresh_test_technical_check_v1"


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
    report = {"schema": CHECK_SCHEMA, "created_utc": now(), "status": "in_progress", "passed": False,
        "policy": spec, "source_run": str(run.resolve()), "identity": manifest["identity"],
        "environment": environment(), "contexts_sha256": file_hash(run / "contexts.jsonl"),
        "predictions_sha256": file_hash(run / "predictions.jsonl"),
        "selection": "longest Original Dev document; all four compression tasks and all five judgments for the first fixed hypothesis",
        "acceptance": "same inputs, valid formats and explicit greedy/memory settings; previous output equality is descriptive only",
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
                if role == "judge":
                    format_valid = parse_label(observed["text"]) is not None
                elif condition.startswith("extractive"):
                    format_valid = select_units(observed["text"], doc["units"])["status"] == "ok"
                else:
                    format_valid = isinstance(observed["text"], str)
                technical = {"same_rendered_prompt": equal["rendered_prompt_sha256"],
                    "same_input_tokens": equal["input_tokens"], "valid_output_format": format_valid,
                    "greedy": observed.get("decoding_policy") == "explicit_greedy_v1",
                    "memory_policy": observed["attention_memory_policy"]["policy_sha256"] == digest(spec)}
                # Keep every newly observed output for inspection, never as a Test
                # context and never as a replacement for the original Dev journal.
                append_jsonl(out / "outputs.jsonl", {"role": role, "condition": condition,
                    "document": longest, "hypothesis": hypothesis if role == "judge" else None,
                    "generation": observed})
                report["checks"].append({"role": role, "condition": condition, "equal": equal,
                    "technical": technical, "passed": all(technical.values()),
                    "output_identical": all(equal.values()), "stored_tokens_sha256": digest(stored["raw_token_ids"]),
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
    report["identical_output_count"] = sum(check["output_identical"] for check in report["checks"])
    report["outputs_sha256"] = file_hash(out / "outputs.jsonl")
    write_json(out / "report.json", report)
    print(f"Dev technical check: {'PASS' if report['passed'] else 'FAIL'}; old outputs identical: {report['identical_output_count']}/9 (informational); report: {out / 'report.json'}")
    return 0 if report["passed"] else 2


def verify_technical_report(report, expected_identity, base_runtime):
    expected = {("judge", condition) for condition in CONDITIONS} | {
                ("compressor", condition) for condition in CONDITIONS[1:]}
    required = {"same_rendered_prompt", "same_input_tokens", "valid_output_format", "greedy", "memory_policy"}
    if (report.get("schema") != CHECK_SCHEMA or report.get("status") != "complete" or
            report.get("passed") is not True or
            len(report.get("checks", [])) != 9 or
            {(check["role"], check["condition"]) for check in report.get("checks", [])} != expected or
            not all(check["passed"] and set(check.get("technical", {})) == required and
                    all(value is True for value in check["technical"].values()) for check in report["checks"])):
        raise ValueError("A passing fresh-Test Dev technical check is required. Run the updated check-dev command.")
    if (report["policy"] != policy_spec() or report["identity"] != expected_identity or
            report["environment"] != environment()):
        raise ValueError("Memory check code, base identity or environment changed.")
    if report["runtime"] != base_runtime:
        raise ValueError("Memory check base runtime differs from the frozen protocol.")


def verify_report(report, manifest, run):
    frozen = read_json(run / "frozen_protocol.json")
    execution = frozen.get("memory_execution", {})
    if execution.get("mode") != "from_start":
        raise ValueError("This runner now requires a fresh uniform Test. Use init-test; do not resume the old Test.")
    if execution.get("policy") != policy_spec() or execution.get("check_report_sha256") != digest(report):
        raise ValueError("Fresh Test memory protocol/check changed.")
    marker = runtime_policy_marker(policy_spec())
    base_runtime = {}
    for role, details in frozen["runtime"].items():
        if details.get("attention_memory_policy") != marker:
            raise ValueError("Frozen model runtime is missing the required memory policy.")
        base_runtime[role] = {key: value for key, value in details.items() if key != "attention_memory_policy"}
    verify_technical_report(report, manifest["identity"], base_runtime)


def initialize_test(config_path, prepared_path, models_path, base_protocol, protocol_output, output, report_path):
    from contract_experiment.data import load_prepared
    from contract_experiment.experiment import initialize

    source = Path(base_protocol)
    protocol, run = Path(protocol_output), Path(output)
    if (protocol.resolve().is_relative_to(source.resolve()) or
            run.resolve().is_relative_to(source.resolve()) or
            protocol.resolve().is_relative_to(run.resolve()) or
            run.resolve().is_relative_to(protocol.resolve())):
        raise ValueError("Base protocol, new protocol and new Test directories must be separate.")
    for path in (protocol, run):
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Destination already contains files; nothing overwritten: {path}")
    base = read_json(source / "protocol.json")
    if base.get("memory_execution"):
        raise ValueError("Use the original greedy Dev protocol as --base-protocol.")
    config = load_config(config_path)
    models = read_json(models_path)
    validate_models(models, config)
    _, prepared = load_prepared(prepared_path)
    expected_identity = identity(config, prepared, models)
    if base["identity"] != expected_identity or base["environment"] != environment():
        raise ValueError("Settings/data/models/base code/environment differ from the original greedy Dev protocol.")
    report = read_json(report_path)
    verify_technical_report(report, expected_identity, base["runtime"])
    spec = policy_spec()
    marker = runtime_policy_marker(spec)
    execution = {"mode": "from_start", "policy": spec, "check_report_sha256": digest(report),
        "base_protocol_path": str(source.resolve()), "base_protocol_sha256": file_hash(source / "protocol.json"),
        "superseded_test": read_json(source / "test_run.json") if (source / "test_run.json").exists() else None,
        "dev_validation": "nine technical outputs, no Gold/accuracy selection; not a full Dev rerun",
        "previous_judgments_reused": 0, "previous_compressed_contexts_reused": 0}
    frozen = {**base, "frozen_utc": now(), "memory_execution": execution,
        "runtime": {role: {**details, "attention_memory_policy": marker} for role, details in base["runtime"].items()},
        "note": "New uniform memory-policy Test protocol; all Test model outputs are generated afresh."}
    protocol.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "protocol.json", protocol / "base_protocol.json")
    write_json(protocol / "memory_check.json", report)
    for name, sha in spec["sources"].items():
        archived = protocol / "memory_extension_source" / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, archived)
        if file_hash(archived) != sha:
            raise ValueError("Memory extension source changed during initialization.")
    write_json(protocol / "protocol.json", frozen)
    # The ordinary initializer still enforces one Test per newly frozen protocol,
    # source/snapshot identity, full Test selection, and a tokenizer-only setup.
    initialize(config_path, prepared_path, models_path, "test", run, protocol=protocol)
    _, manifest, *_ = load_run(run)
    with run_lock(run):
        verify_report(report, manifest, run)
        adopt(run, manifest, report)
    print(f"Uniform memory policy frozen: {protocol.resolve()}")
    print("Fresh Test initialized: 0 previous judgments and 0 previous compressed contexts reused.")


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
        if (record.get("mode") != "from_start" or record.get("base_identity") != manifest["identity"] or
                record["journal_boundaries"]["predictions.jsonl"] != 0 or
                record["journal_boundaries"]["contexts.jsonl"] != manifest["document_count"] or
                record["policy"] != policy_spec() or record["check_report_sha256"] != digest(report)):
            raise ValueError("Recorded memory policy/check changed; do not mix execution policies.")
        for name, sha in record["policy"]["sources"].items():
            if file_hash(run / "memory_extension_source" / name) != sha:
                raise ValueError("Archived memory extension source changed.")
        verify_new_rows(run, record, repair=resume)
        return record
    journals = {name: read_jsonl(run / name, repair=resume) for name in ("contexts.jsonl", "predictions.jsonl")}
    originals = journals["contexts.jsonl"]
    if (journals["predictions.jsonl"] or len(originals) != manifest["document_count"] or
            any(row.get("condition") != "original" or row.get("generation") is not None for row in originals) or
            {row["document"] for row in originals} != set(manifest["document_keys"])):
        raise ValueError("Fresh Test must have only initialized Original contexts and zero judgments; use a new run.")
    record = {"mode": "from_start", "created_utc": now(), "base_identity": manifest["identity"], "policy": policy_spec(),
        "check_report_sha256": digest(report), "check_report": report,
        "journal_boundaries": {name: len(rows) for name, rows in journals.items()},
        "journal_prefix_digests": {name: digest(rows) for name, rows in journals.items()},
        "previous_judgments_reused": 0, "previous_compressed_contexts_reused": 0,
        "note": "Uniform execution policy from the first Test model call. Original contexts are raw data, not reused model outputs."}
    for name in record["policy"]["sources"]:
        target = run / "memory_extension_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(path, record)
    append_jsonl(run / "events.jsonl", {"event": "memory_policy_adopted", "time": now(),
        "policy_sha256": digest(record["policy"]), "mode": "from_start", "initial_rows": record["journal_boundaries"]})
    return record


def add_report_note(run):
    record = read_json(run / "runtime_memory_policy.json")
    for analysis_path in (run / "results" / "analysis.json", run / "results" / "baseline" / "analysis.json"):
        if analysis_path.exists():
            analysis = read_json(analysis_path)
            analysis["attention_memory_execution"] = {
                "policy_id": record["policy"]["id"], "policy_sha256": digest(record["policy"]),
                "execution_mode": "from_start", "initial_journal_rows": record["journal_boundaries"],
                "previous_judgments_reused": 0, "previous_compressed_contexts_reused": 0,
                "provenance_file": "runtime_memory_policy.json",
                "technical_check_scope": "9 Dev outputs on the longest Original Dev document; old output equality is informational",
                "timing_note": "Same memory policy throughout this Test; operation counts depend on input length"}
            write_json(analysis_path, analysis)
    for report_path in (run / "results" / "report.md", run / "results" / "baseline" / "report.md"):
        if report_path.exists():
            text = report_path.read_text(encoding="utf-8")
            marker = "## Attention memory execution record"
            if marker not in text:
                text += ("\n\n" + marker + "\n\nAll Test compression and judgments were generated under the same frozen memory policy from the first model call. "
                    "No prior predictions or compressed contexts were reused. Full inputs and base model settings were preserved. "
                    "Old Dev output equality was descriptive, not an acceptance criterion. Read runtime_memory_policy.json for source hashes and technical validation. "
                    "Short/long inputs follow the same fixed dispatch rule; their operation counts and timing naturally differ.\n")
                report_path.write_text(text, encoding="utf-8")


def execute(arguments):
    from contract_experiment import cli, experiment

    args = cli.parser().parse_args(arguments)
    if args.command not in {"judge", "compress", "diagnose", "analyze"}:
        raise ValueError("Use init-test for fresh initialization, then judge/compress/diagnose/analyze with this runner.")
    run, manifest, config, models, dataset = load_run(args.run)
    if manifest["split"] != "test":
        raise ValueError("This recovery runner is for Test; use check-dev for its Dev comparison.")
    record_path = run / "runtime_memory_policy.json"
    if not record_path.exists():
        raise ValueError("Memory initialization is incomplete. Use a fresh init-test run.")
    report = read_json(record_path)["check_report"]
    verify_report(report, manifest, run)
    with run_lock(run):
        adopt(run, manifest, report, resume=getattr(args, "resume", False))
    original = experiment.LocalModel
    experiment.LocalModel = memory_model(original, policy_spec(), stamp_runtime=True)
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
    parser = argparse.ArgumentParser(description="Fresh Test with one frozen full-context memory policy from its first model call.", add_help=False)
    parser.add_argument("-h", "--help", action="store_true", help="show command help")
    parser.add_argument("--check-report", default="runs/memory_check_fresh/report.json")
    options, remaining = parser.parse_known_args(arguments)
    try:
        check_arguments = [argument for argument in remaining if argument != "--debug"]
        if options.help and check_arguments:
            check_arguments.append("--help")
        if check_arguments and check_arguments[0] == "check-dev":
            check = argparse.ArgumentParser(description="Nine technical output checks; old output equality is descriptive, not a gate.")
            check.add_argument("--run", default="runs/dev_greedy")
            check.add_argument("--out", default="runs/memory_check_fresh")
            args = check.parse_args(check_arguments[1:])
            return check_dev(args.run, args.out)
        if check_arguments and check_arguments[0] == "init-test":
            init = argparse.ArgumentParser(description="Freeze a new memory protocol and initialize an entirely fresh Test; tokenizers only.")
            init.add_argument("--config", default="config.toml")
            init.add_argument("--prepared", default="prepared")
            init.add_argument("--models", default="model_lock.json")
            init.add_argument("--base-protocol", default="protocol_greedy")
            init.add_argument("--protocol-out", default="protocol_memory_safe")
            init.add_argument("--out", default="runs/test_memory_safe")
            args = init.parse_args(check_arguments[1:])
            initialize_test(args.config, args.prepared, args.models, args.base_protocol,
                            args.protocol_out, args.out, options.check_report)
            return 0
        if options.help or not remaining:
            parser.print_help()
            print("Use check-dev, then init-test, then judge/compress/diagnose/analyze --run runs/test_memory_safe.")
            return 0
        return execute(remaining)
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
