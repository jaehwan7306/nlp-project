"""Explicit, resumable stages; one GPU model at a time."""

import shutil
import time
from collections import Counter
from pathlib import Path

from . import CONDITIONS
from .data import load_prepared
from .prompts import compression_messages, judge_messages, parse_label, prompt_spec, select_units
from .runtime import LocalModel, load_tokenizer, render, token_count
from .settings import ROOT, code_hashes, environment, identity, load_config, load_run, validate_models
from .storage import (append_jsonl, digest, file_hash, now, read_json, read_jsonl,
                      run_lock, unique_index, write_csv, write_json)


def initialize(config_path, prepared_path, models_path, split, output, documents=None, protocol=None):
    config = load_config(config_path)
    models = read_json(models_path)
    validate_models(models, config)
    dataset, prepared = load_prepared(prepared_path)
    selected = sorted((d for d in dataset["documents"] if d["split"] == split), key=lambda d: d["id"])
    full_count = len(selected)
    if documents is not None:
        if split == "test" or documents < 1 or documents > full_count:
            raise ValueError("--documents is only for a nonempty Dev subset, within its document count.")
        selected = sorted(sorted(selected, key=lambda d: (-len(d["text"]), d["id"]))[:documents], key=lambda d: d["id"])
    if not selected:
        raise ValueError("No documents available for this split.")
    run_identity = identity(config, prepared, models)
    frozen = None
    if split == "test":
        if protocol is None:
            raise ValueError("Test requires --protocol from a completed full Dev technical check.")
        frozen = read_json(Path(protocol) / "protocol.json")
        if frozen["identity"] != run_identity:
            raise ValueError("Test settings/data/models/code differ from the frozen Dev protocol.")
        if frozen["environment"] != environment():
            raise ValueError("Python/platform/package versions differ from the frozen Dev environment.")
        if (Path(protocol) / "test_run.json").exists():
            raise ValueError("This protocol already started Test. Resume the existing run instead of initializing another.")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Run directory already contains files. Use --resume on its stage, or choose a new Dev directory.")
    # Tokenizer only: Original length is measured before any model inference.
    counter = load_tokenizer(config, models, "judge")
    originals = []
    for doc in selected:
        count = token_count(counter, doc["text"])
        if count == 0:
            raise ValueError(f"Empty tokenized source: {doc['key']}")
        originals.append({"document": doc["key"], "condition": "original", "status": "ok",
            "text": doc["text"], "context_tokens": count, "original_tokens": count,
            "retention": 1.0, "compression_seconds": 0.0, "generation": None,
            "selected_ids": None, "source_ranges": None, "error": None})
    out.mkdir(parents=True, exist_ok=True)
    snapshots = {"config.json": config, "models.json": models, "prepared_manifest.json": prepared,
                 "prompts.json": prompt_spec(), "environment.json": environment(),
                 "dataset.json": {"hypotheses": dataset["hypotheses"], "documents": selected}}
    if frozen is not None:
        snapshots["frozen_protocol.json"] = frozen
    for name, payload in snapshots.items():
        write_json(out / name, payload)
    for relative in code_hashes():
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    manifest = {"created_utc": now(), "split": split, "conditions": list(CONDITIONS),
        "document_keys": [d["key"] for d in selected], "full_split_count": full_count,
        "document_count": len(selected), "hypotheses_per_document": 17,
        "expected_contexts": 5 * len(selected), "expected_judgments": 5 * 17 * len(selected),
        "identity": run_identity, "code_hashes": code_hashes(),
        "snapshot_hashes": {name: file_hash(out / name) for name in snapshots}}
    write_json(out / "manifest.json", manifest)
    for row in originals:
        append_jsonl(out / "contexts.jsonl", row)
    if split == "test":
        # Exclusive creation also prevents two simultaneous initializations.
        marker = Path(protocol) / "test_run.json"
        with marker.open("x", encoding="utf-8") as handle:
            import json
            json.dump({"run_directory": str(out.resolve()), "started_utc": now(),
                       "identity": run_identity}, handle, ensure_ascii=False, indent=2)
    print(f"Initialized {split}: {len(selected)} documents × 17 hypotheses × 5 conditions = {manifest['expected_judgments']} judgments.")


def model_details(run, model):
    details = model.details
    comparable = {k: v for k, v in details.items() if k != "load_seconds"}
    path = run / f"runtime_{model.role}.json"
    if path.exists():
        previous = {k: v for k, v in read_json(path).items() if k != "load_seconds"}
        if digest(previous) != digest(comparable):
            raise ValueError("GPU/runtime/template settings changed during resume.")
    protocol = run / "frozen_protocol.json"
    if protocol.exists():
        expected = read_json(protocol)["runtime"][model.role]
        if digest(expected) != digest(comparable):
            raise ValueError("Test GPU/runtime/template settings differ from the frozen Dev execution.")
    if not path.exists():
        write_json(path, details)
    append_jsonl(run / "events.jsonl", {"event": "model_loaded", "time": now(), "role": model.role,
                                        "seconds": details["load_seconds"]})


def check_environment(run):
    if read_json(run / "environment.json") != environment():
        raise ValueError("Python/platform/package versions changed since run initialization.")


def compress(run_path, resume=False):
    run, manifest, config, models, dataset = load_run(run_path)
    with run_lock(run):
        check_environment(run)
        rows = read_jsonl(run / "contexts.jsonl", repair=resume)
        index = unique_index(rows, ("document", "condition"))
        if not resume and any(row["condition"] != "original" for row in rows):
            raise ValueError("Compression has existing rows. Use --resume to skip completed items.")
        tasks = [(doc, condition) for doc in dataset["documents"] for condition in CONDITIONS[1:]
                 if (doc["key"], condition) not in index]
        if not tasks:
            print("All compression records already exist; no model loaded.")
            return
        counter = load_tokenizer(config, models, "judge")
        model = LocalModel(config, models, "compressor")
        try:
            model_details(run, model)
            for number, (doc, condition) in enumerate(tasks, 1):
                print(f"[compress {number}/{len(tasks)}] {doc['key']} {condition}", flush=True)
                start = time.perf_counter()
                generated = model.generate(compression_messages(condition, doc["units"]))
                if condition.startswith("extractive"):
                    selected = select_units(generated["text"], doc["units"])
                else:
                    selected = {"status": "ok", "error": None, "text": generated["text"],
                                "selected_ids": None, "source_ranges": None,
                                "empty_output": not generated["text"],
                                "full_source_copy": generated["text"].strip() == doc["text"].strip()}
                original_tokens = index[(doc["key"], "original")]["original_tokens"]
                context_tokens = token_count(counter, selected["text"]) if selected["status"] == "ok" else None
                row = {"document": doc["key"], "condition": condition, **selected,
                       "context_tokens": context_tokens, "original_tokens": original_tokens,
                       "retention": context_tokens / original_tokens if context_tokens is not None else None,
                       "compression_seconds": time.perf_counter() - start, "generation": generated}
                append_jsonl(run / "contexts.jsonl", row)
        finally:
            model.close()
    print("Compression complete. Run judge next.")


def failed_prediction(doc, condition, hid):
    return {"document": doc["key"], "condition": condition, "hypothesis": hid,
            "gold": doc["annotations"][hid]["choice"], "prediction": None,
            "status": "compression_failed", "judgment_seconds": 0.0, "generation": None}


def judge(run_path, conditions=None, resume=False):
    run, manifest, config, models, dataset = load_run(run_path)
    conditions = tuple(conditions or CONDITIONS)
    if len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Unknown or duplicate judgment condition.")
    with run_lock(run):
        check_environment(run)
        contexts = unique_index(read_jsonl(run / "contexts.jsonl"), ("document", "condition"))
        existing = read_jsonl(run / "predictions.jsonl", repair=resume)
        predictions = unique_index(existing, ("document", "condition", "hypothesis"))
        if not resume and any(row["condition"] in conditions for row in existing):
            raise ValueError("Predictions already exist for a requested condition. Use --resume.")
        tasks = []
        # Evaluate the entire Original baseline before any compressed condition.
        for condition in conditions:
            for doc in dataset["documents"]:
                if (doc["key"], condition) not in contexts:
                    raise ValueError(f"Missing compression: {doc['key']} {condition}. Run compress first.")
                for hid in dataset["hypotheses"]:
                    if (doc["key"], condition, hid) not in predictions:
                        tasks.append((doc, condition, hid))
        if not tasks:
            print("Requested predictions already exist; no model loaded.")
            return
        model = None
        try:
            for number, (doc, condition, hid) in enumerate(tasks, 1):
                print(f"[judge {number}/{len(tasks)}] {doc['key']} {condition} {hid}", flush=True)
                context = contexts[(doc["key"], condition)]
                if context["status"] != "ok":
                    row = failed_prediction(doc, condition, hid)
                else:
                    if model is None:
                        model = LocalModel(config, models, "judge")
                        model_details(run, model)
                    start = time.perf_counter()
                    generated = model.generate(judge_messages(context["text"], dataset["hypotheses"][hid]))
                    label = parse_label(generated["text"])
                    row = {"document": doc["key"], "condition": condition, "hypothesis": hid,
                           "gold": doc["annotations"][hid]["choice"], "prediction": label,
                           "status": "ok" if label else "parse_failed",
                           "judgment_seconds": time.perf_counter() - start, "generation": generated}
                append_jsonl(run / "predictions.jsonl", row)
        finally:
            if model is not None:
                model.close()
    print("Judgment complete. Dev: diagnose; Test: diagnose and analyze.")


def validated_results(run, dataset, conditions=CONDITIONS):
    rows = read_jsonl(run / "predictions.jsonl")
    index = unique_index(rows, ("document", "condition", "hypothesis"))
    docs = {doc["key"]: doc for doc in dataset["documents"]}
    all_keys = {(key, condition, hid) for key in docs for condition in CONDITIONS for hid in dataset["hypotheses"]}
    if set(index) - all_keys:
        raise ValueError("Unexpected prediction keys.")
    required = {(key, condition, hid) for key in docs for condition in conditions for hid in dataset["hypotheses"]}
    if required - set(index):
        raise ValueError(f"Missing {len(required - set(index))} predictions. Resume judge before analyzing/freezing.")
    for key, row in index.items():
        if row["gold"] != docs[key[0]]["annotations"][key[2]]["choice"]:
            raise ValueError(f"Gold label mismatch in result: {key}")
        if row["status"] not in {"ok", "parse_failed", "compression_failed"}:
            raise ValueError(f"Unexpected prediction status: {key}")
        if (row["status"] == "ok") != (row["prediction"] in ("Entailment", "Contradiction", "NotMentioned")):
            raise ValueError(f"Prediction/status mismatch: {key}")
        if row["status"] != "ok" and row["prediction"] is not None:
            raise ValueError(f"A failed prediction must be null: {key}")
    return [index[key] for key in sorted(required)]


def diagnose(run_path):
    run, manifest, config, models, dataset = load_run(run_path)
    with run_lock(run):
        contexts = read_jsonl(run / "contexts.jsonl")
        unique_index(contexts, ("document", "condition"))
        predictions = read_jsonl(run / "predictions.jsonl")
        unique_index(predictions, ("document", "condition", "hypothesis"))
        summary = {"accuracy_computed": False, "context_count": len(contexts),
                   "contexts_sha256": file_hash(run / "contexts.jsonl"),
                   "predictions_sha256": file_hash(run / "predictions.jsonl") if (run / "predictions.jsonl").exists() else None,
                   "expected_context_count": manifest["expected_contexts"],
                   "prediction_count": len(predictions), "expected_prediction_count": manifest["expected_judgments"],
                   "conditions": {}}
        lengths = []
        for condition in CONDITIONS:
            rows = [row for row in contexts if row["condition"] == condition]
            judgments = [row for row in predictions if row["condition"] == condition]
            summary["conditions"][condition] = {
                "contexts": len(rows), "compression_failed": sum(r["status"] != "ok" for r in rows),
                "stop_reasons": dict(Counter(r["generation"]["stop_reason"] for r in rows if r["generation"])),
                "cap_reached": sum(bool(r["generation"] and r["generation"]["cap_reached"]) for r in rows),
                "dialogue_flags": sum(bool(r["generation"] and r["generation"]["dialogue_flag"]) for r in rows),
                "full_source_copies": sum(bool(r.get("full_source_copy")) for r in rows),
                "empty_outputs": sum(r["status"] == "ok" and not r["text"] for r in rows),
                "duplicate_id_outputs": sum(bool(r.get("duplicate_ids")) for r in rows),
                "judgments": len(judgments), "parsing_failures": sum(r["status"] == "parse_failed" for r in judgments),
                "judgment_stop_reasons": dict(Counter(r["generation"]["stop_reason"] for r in judgments if r["generation"])),
            }
            lengths.extend({"document": r["document"], "condition": condition, "status": r["status"],
                "original_tokens": r["original_tokens"], "context_tokens": r["context_tokens"],
                "retention": r["retention"], "selected_unit_count": len(r["selected_ids"]) if r.get("selected_ids") is not None else ""}
                for r in rows)
        out = run / "diagnostics"
        write_json(out / "technical_summary.json", summary)
        write_csv(out / "lengths.csv", lengths, ["document", "condition", "status", "original_tokens", "context_tokens", "retention", "selected_unit_count"])
    print(f"Technical diagnostics (no accuracy): {out.resolve()}")


def freeze(run_path, output):
    run, manifest, config, models, dataset = load_run(run_path)
    if manifest["split"] != "dev" or manifest["document_count"] != manifest["full_split_count"]:
        raise ValueError("Freeze requires a full prepared Dev run, not a small preflight subset.")
    with run_lock(run):
        check_environment(run)
        validated_results(run, dataset)
        contexts = unique_index(read_jsonl(run / "contexts.jsonl"), ("document", "condition"))
        expected = {(doc["key"], condition) for doc in dataset["documents"] for condition in CONDITIONS}
        if set(contexts) != expected:
            raise ValueError("Dev context records are incomplete or have unexpected keys.")
        if not (run / "diagnostics" / "technical_summary.json").exists():
            raise ValueError("Run diagnose and inspect its technical outputs before freeze.")
        diagnostics = read_json(run / "diagnostics" / "technical_summary.json")
        if diagnostics["contexts_sha256"] != file_hash(run / "contexts.jsonl") or diagnostics["predictions_sha256"] != file_hash(run / "predictions.jsonl"):
            raise ValueError("Diagnostics are stale. Run diagnose again after completing all stages.")
        roles = {}
        for role in ("compressor", "judge"):
            roles[role] = {k: v for k, v in read_json(run / f"runtime_{role}.json").items() if k != "load_seconds"}
        out = Path(output)
        if out.exists() and any(out.iterdir()):
            raise ValueError("Protocol already exists. Never overwrite a frozen protocol.")
        write_json(out / "protocol.json", {"frozen_utc": now(), "identity": manifest["identity"],
            "environment": read_json(run / "environment.json"), "runtime": roles,
            "dev_document_count": manifest["document_count"], "expected_test_judgments": 5 * 17 * read_json(run / "prepared_manifest.json")["counts"]["test"],
            "technical_diagnostics": read_json(run / "diagnostics" / "technical_summary.json"),
            "prompts": prompt_spec(), "config": config, "code_hashes": manifest["code_hashes"],
            "note": "Technical protocol freeze; no accuracy threshold and no mandatory human fidelity gate."})
    print(f"Protocol frozen: {out.resolve()}")


def inspect_inputs(config_path, prepared_path, models_path, output):
    config, models = load_config(config_path), read_json(models_path)
    validate_models(models, config)
    dataset, _ = load_prepared(prepared_path)
    comp = load_tokenizer(config, models, "compressor")
    judge_tokenizer = load_tokenizer(config, models, "judge")
    rows = []
    for doc in dataset["documents"]:
        comp_prompt_tokens = max(token_count(comp, render(comp, compression_messages(c, doc["units"]))) for c in CONDITIONS[1:])
        original_judge_tokens = max(token_count(judge_tokenizer, render(judge_tokenizer, judge_messages(doc["text"], h), judge=True))
                                   for h in dataset["hypotheses"].values())
        limit, generation = config["runtime"]["context_limit"], config["generation"]
        rows.append({"document": doc["key"], "original_qwen25_tokens": token_count(comp, doc["text"]),
            "original_qwen3_tokens": token_count(judge_tokenizer, doc["text"]), "unit_count": len(doc["units"]),
            "gap_units": sum(u["kind"] == "gap" for u in doc["units"]),
            "max_compression_prompt_tokens": comp_prompt_tokens, "max_original_judge_prompt_tokens": original_judge_tokens,
            "compression_fits": comp_prompt_tokens + generation["compressor_max_new_tokens"] <= limit,
            "original_judge_fits": original_judge_tokens + generation["judge_max_new_tokens"] <= limit})
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Inspection output exists; choose a new output directory.")
    write_csv(out / "input_stats.csv", rows, list(rows[0]))
    write_json(out / "summary.json", {"tokenizer_only": True, "weights_loaded": False,
        "documents": len(rows), "over_context_limit": [r["document"] for r in rows if not r["compression_fits"] or not r["original_judge_fits"]],
        "longest_dev_documents": [r["document"] for r in sorted((r for r in rows if r["document"].startswith("dev:")),
                                    key=lambda r: r["max_compression_prompt_tokens"], reverse=True)[:3]],
        "max_compression_prompt_tokens": max(r["max_compression_prompt_tokens"] for r in rows),
        "max_original_judge_prompt_tokens": max(r["max_original_judge_prompt_tokens"] for r in rows)})
    print(f"Tokenizer-only inspection: {out.resolve()}. Check summary.json before GPU inference.")
