"""Immutable question-run snapshots and a validated reference to completed Main."""

import json
import os
import shutil
from pathlib import Path

from contract_experiment import CONDITIONS
from contract_experiment.experiment import validated_results
from contract_experiment.runtime import load_tokenizer, render, token_count
from contract_experiment.settings import ROOT, code_hashes, environment, load_run as load_main_run
from contract_experiment.storage import digest, file_hash, now, read_json, read_jsonl, unique_index, write_json
from memory_support.attention import policy_spec, runtime_policy_marker
from .prompts import compression_messages, prompt_spec

EXPERIMENT = "analysis1_q17_shared"


def question_sources():
    files = [ROOT / "run_question_analysis.py", *sorted((ROOT / "question_support").glob("*.py"))]
    return {file.relative_to(ROOT).as_posix(): file_hash(file) for file in files}


def inspect_main(path, require_environment=False):
    run, manifest, config, models, dataset = load_main_run(path)
    if manifest["split"] != "test" or manifest.get("experiment") or manifest["document_count"] != manifest["full_split_count"]:
        raise ValueError("--main-run must be the completed, full, question-free Main Test.")
    contexts = unique_index(read_jsonl(run / "contexts.jsonl"), ("document", "condition"))
    predictions = unique_index(validated_results(run, dataset), ("document", "condition", "hypothesis"))
    expected = {(doc["key"], condition) for doc in dataset["documents"] for condition in CONDITIONS}
    if set(contexts) != expected:
        raise ValueError("Main contexts are incomplete or contain unexpected keys.")
    frozen = read_json(run / "frozen_protocol.json")
    memory = read_json(run / "runtime_memory_policy.json")
    spec = policy_spec()
    if (memory.get("mode") != "from_start" or frozen.get("memory_execution", {}).get("mode") != "from_start" or
            memory["policy"] != spec or memory["journal_boundaries"]["predictions.jsonl"] != 0):
        raise ValueError("Main must use the same frozen memory policy from its first model call.")
    diagnostics = read_json(run / "diagnostics/technical_summary.json")
    if diagnostics["contexts_sha256"] != file_hash(run / "contexts.jsonl") or diagnostics["predictions_sha256"] != file_hash(run / "predictions.jsonl"):
        raise ValueError("Main diagnostics are stale.")
    for role in ("compressor", "judge"):
        actual = {k: v for k, v in read_json(run / f"runtime_{role}.json").items() if k != "load_seconds"}
        if actual != frozen["runtime"][role] or actual.get("attention_memory_policy") != runtime_policy_marker(spec):
            raise ValueError(f"Main {role} runtime does not match its frozen memory protocol.")
    for row in [*contexts.values(), *predictions.values()]:
        generation = row.get("generation")
        if generation and (generation.get("decoding_policy") != "explicit_greedy_v1" or
                generation.get("attention_memory_policy", {}).get("policy_sha256") != digest(spec)):
            raise ValueError("Main contains output from a different decoding or memory policy.")
    for name, sha in spec["sources"].items():
        if file_hash(run / "memory_extension_source" / name) != sha:
            raise ValueError("The installed memory implementation differs from Main; restore the Main version.")
    if require_environment and read_json(run / "environment.json") != environment():
        raise ValueError("Use exactly the same Python, hardware runtime and package versions as Main.")
    names = ("manifest.json", "dataset.json", "prepared_manifest.json", "config.json", "models.json",
             "prompts.json", "environment.json", "frozen_protocol.json", "runtime_memory_policy.json",
             "runtime_compressor.json", "runtime_judge.json", "contexts.jsonl", "predictions.jsonl")
    return {"run": run, "manifest": manifest, "config": config, "models": models, "dataset": dataset,
        "contexts": contexts, "predictions": predictions, "frozen": frozen, "memory": spec,
        "hashes": {name: file_hash(run / name) for name in names}}


def question_protocol(main):
    prompts = prompt_spec(main["dataset"]["hypotheses"])
    sources = {**code_hashes(), **main["memory"]["sources"], **question_sources()}
    identity = {**main["manifest"]["identity"], "code": digest(sources), "prompts": digest(prompts),
        "main_reference": digest(main["hashes"]), "memory_policy": digest(main["memory"])}
    marker = {"mode": "all_17_shared", "protocol_identity": digest(identity)}
    return {"schema": EXPERIMENT, "identity": identity, "base_identity": main["manifest"]["identity"],
        "code_hashes": sources, "source_main_hashes": main["hashes"], "prompts": prompts,
        "memory_policy": main["memory"], "question_context_policy": marker,
        "runtime": {role: {**details, "question_context_policy": marker} for role, details in main["frozen"]["runtime"].items()},
        "note": "Same Main settings. Only compressor input adds all 17 hypotheses. Gold labels/evidence are never provided to compression."}


def write_rows(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def initialize(main, documents, split, output, dev_source=None, preflight_receipt=None):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory already contains files. Use --resume for its stage, or choose a new directory.")
    protocol = question_protocol(main)
    originals, reused = [], []
    if split == "test":
        originals = [main["contexts"][(doc["key"], "original")] for doc in documents]
        reused = [main["predictions"][(doc["key"], "original", h)] for doc in documents for h in main["dataset"]["hypotheses"]]
    else:
        counter = load_tokenizer(main["config"], main["models"], "judge")
        for doc in documents:
            count = token_count(counter, doc["text"])
            if count == 0:
                raise ValueError(f"Empty original context: {doc['key']}")
            originals.append({"document": doc["key"], "condition": "original", "status": "ok", "text": doc["text"],
                "context_tokens": count, "original_tokens": count, "retention": 1.0, "compression_seconds": 0.0,
                "generation": None, "selected_ids": None, "source_ranges": None, "error": None})
    out.mkdir(parents=True, exist_ok=True)
    lineage = {"created_utc": now(), "main_run_relative": os.path.relpath(main["run"].resolve(), out.resolve()),
        "main_run_at_creation": str(main["run"].resolve()), "source_main_hashes": main["hashes"],
        "original_contexts_digest": digest(originals), "reused_original_predictions_digest": digest(reused),
        "reused_original_judgments": len(reused), "reused_question_free_compressed_contexts": 0,
        "preflight_receipt": preflight_receipt, "dev_source": dev_source}
    snapshots = {"config.json": main["config"], "models.json": main["models"],
        "prepared_manifest.json": read_json(main["run"] / "prepared_manifest.json"),
        "dataset.json": {"hypotheses": main["dataset"]["hypotheses"], "documents": documents},
        "environment.json": read_json(main["run"] / "environment.json"), "prompts.json": protocol["prompts"],
        "frozen_protocol.json": protocol, "lineage.json": lineage}
    for name, value in snapshots.items():
        write_json(out / name, value)
    for relative in protocol["code_hashes"]:
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write_rows(out / "contexts.jsonl", originals)
    write_rows(out / "predictions.jsonl", reused)
    n = len(documents)
    manifest = {"created_utc": now(), "experiment": EXPERIMENT, "split": split,
        "conditions": list(CONDITIONS), "document_keys": [doc["key"] for doc in documents], "document_count": n,
        "full_split_count": main["manifest"]["document_count"] if split == "test" else dev_source["full_document_count"],
        "hypotheses_per_document": 17, "expected_contexts": 5 * n, "expected_judgments": 85 * n,
        "expected_new_compressions": 4 * n, "expected_new_judgments": (68 if split == "test" else 85) * n,
        "reused_original_judgments": len(reused), "identity": protocol["identity"], "base_identity": protocol["base_identity"],
        "code_hashes": protocol["code_hashes"], "snapshot_hashes": {name: file_hash(out / name) for name in snapshots}}
    write_json(out / "manifest.json", manifest)
    print(f"Initialized Q17 {split}: {n} documents; new compression={4*n}; new judgments={manifest['expected_new_judgments']}; reused Original={len(reused)}.")


def load_question_run(path, repair=False):
    run = Path(path)
    manifest = read_json(run / "manifest.json")
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Use a dedicated Q17 run, not a Main run.")
    for name, sha in manifest["snapshot_hashes"].items():
        if file_hash(run / name) != sha:
            raise ValueError(f"Question run snapshot changed: {name}")
    lineage = read_json(run / "lineage.json")
    main = inspect_main((run / lineage["main_run_relative"]).resolve())
    protocol = read_json(run / "frozen_protocol.json")
    expected = question_protocol(main)
    if protocol != expected or manifest["identity"] != expected["identity"] or manifest["code_hashes"] != expected["code_hashes"]:
        raise ValueError("Question prompts, settings, source Main or executable source changed. Do not mix runs.")
    for relative, sha in manifest["code_hashes"].items():
        if file_hash(run / "source" / relative) != sha:
            raise ValueError(f"Archived question source changed: {relative}")
    if lineage["source_main_hashes"] != main["hashes"]:
        raise ValueError("The reference Main changed.")
    config, models, dataset = (read_json(run / name) for name in ("config.json", "models.json", "dataset.json"))
    if (config != main["config"] or models != main["models"] or dataset["hypotheses"] != main["dataset"]["hypotheses"] or
            manifest["document_keys"] != [d["key"] for d in dataset["documents"]] or
            manifest["document_count"] != len(dataset["documents"]) or manifest["conditions"] != list(CONDITIONS) or
            manifest["split"] not in {"dev", "test"} or
            any(doc["split"] != manifest["split"] for doc in dataset["documents"]) or
            manifest["expected_contexts"] != 5 * len(dataset["documents"]) or
            manifest["expected_judgments"] != 85 * len(dataset["documents"])):
        raise ValueError("Question dataset/configuration/counts do not match Main and the frozen manifest.")
    if manifest["split"] == "test" and dataset != main["dataset"]:
        raise ValueError("Q17 Test must use exactly the Main documents and annotations.")
    contexts = read_jsonl(run / "contexts.jsonl", repair=repair)
    predictions = read_jsonl(run / "predictions.jsonl", repair=repair)
    ci = unique_index(contexts, ("document", "condition"))
    pi = unique_index(predictions, ("document", "condition", "hypothesis"))
    docs = manifest["document_keys"]
    if set(ci) - {(d, c) for d in docs for c in CONDITIONS} or set(pi) - {(d, c, h) for d in docs for c in CONDITIONS for h in dataset["hypotheses"]}:
        raise ValueError("Unexpected context or judgment keys.")
    originals = [ci[(d, "original")] for d in docs]
    if digest(originals) != lineage["original_contexts_digest"]:
        raise ValueError("Original contexts changed.")
    if manifest["split"] == "test":
        copied = [pi[(d, "original", h)] for d in docs for h in dataset["hypotheses"]]
        if digest(copied) != lineage["reused_original_predictions_digest"]:
            raise ValueError("Reused Main Original judgments changed.")
    docmap = {doc["key"]: doc for doc in dataset["documents"]}
    for row in predictions:
        if (row["gold"] != docmap[row["document"]]["annotations"][row["hypothesis"]]["choice"] or
                row["status"] not in {"ok", "parse_failed", "compression_failed"} or
                (row["status"] == "ok") != (row["prediction"] in {"Entailment", "Contradiction", "NotMentioned"}) or
                (row["status"] != "ok" and row["prediction"] is not None)):
            raise ValueError("Invalid question-run prediction/Gold/status record.")
    for row in contexts + predictions:
        generation = row.get("generation")
        if generation is None or (manifest["split"] == "test" and row["condition"] == "original"):
            continue
        if (generation.get("question_context_policy") != protocol["question_context_policy"] or
                generation.get("attention_memory_policy", {}).get("policy_sha256") != digest(main["memory"]) or
                generation.get("decoding_policy") != "explicit_greedy_v1"):
            raise ValueError("Output lacks the frozen Q17/greedy/Main memory policy; use run_question_analysis.py.")
    return run, manifest, config, models, dataset


def preflight_documents(main, dev_path, number):
    dev, manifest, config, models, dataset = load_main_run(dev_path)
    if (manifest["split"] != "dev" or manifest["identity"] != main["manifest"]["identity"] or
            dataset["hypotheses"] != main["dataset"]["hypotheses"] or number < 1 or number > len(dataset["documents"])):
        raise ValueError("Use a matching Dev run and a valid --documents count.")
    tokenizer = load_tokenizer(config, models, "compressor")
    lengths = {doc["key"]: max(token_count(tokenizer, render(tokenizer,
        compression_messages(c, doc["units"], dataset["hypotheses"]))) for c in CONDITIONS[1:]) for doc in dataset["documents"]}
    selected = sorted(sorted(dataset["documents"], key=lambda d: (-lengths[d["key"]], d["id"]))[:number], key=lambda d: d["id"])
    receipt = {"source_run": str(dev.resolve()), "source_manifest_sha256": file_hash(dev / "manifest.json"),
        "source_dataset_sha256": file_hash(dev / "dataset.json"), "full_document_count": len(dataset["documents"]),
        "selection": "longest rendered Q17 compression inputs; no accuracy selection",
        "selected_input_tokens": {d["key"]: lengths[d["key"]] for d in selected}}
    return selected, receipt
