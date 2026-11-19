"""Initialize and verify a separate Q8 run with read-only reference runs."""

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from contract_experiment import CONDITIONS
from contract_experiment.experiment import validated_results
from contract_experiment.settings import ROOT, code_hashes
from contract_experiment.storage import digest, file_hash, now, read_json, read_jsonl, unique_index, write_json
from memory_support.attention import policy_spec, runtime_policy_marker
from question_support.state import inspect_main, load_question_run, question_sources, write_rows

from .assignment import EXPERIMENT, balanced_assignment, prompt_spec


CONTEXT_KEY = ("document", "condition")
PREDICTION_KEY = (*CONTEXT_KEY, "hypothesis")
JOURNALS = ("contexts.jsonl", "predictions.jsonl")


def source_hashes(memory):
    files = [ROOT / "run_question_subset.py", *sorted((ROOT / "question_subset").glob("*.py"))]
    return {**code_hashes(), **memory["sources"], **question_sources(),
            **{path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}}


def inspect_references(main_path, q17_path):
    main = inspect_main(Path(main_path).resolve())
    q17, manifest, config, models, dataset = load_question_run(Path(q17_path).resolve())
    lineage = read_json(q17 / "lineage.json")
    if (manifest["split"] != "test" or dataset != main["dataset"] or
            list(dataset["hypotheses"]) != list(main["dataset"]["hypotheses"]) or
            lineage["source_main_hashes"] != main["hashes"] or
            config != main["config"] or models != main["models"] or
            file_hash(q17 / "environment.json") != main["hashes"]["environment.json"]):
        raise ValueError("Q17 must be the completed Test for the supplied Q0 with identical data and settings.")
    protocol = read_json(q17 / "frozen_protocol.json")
    for role, expected in protocol["runtime"].items():
        actual = read_json(q17 / f"runtime_{role}.json")
        actual.pop("load_seconds")
        if actual != expected:
            raise ValueError(f"Q17 {role} runtime differs from the frozen protocol.")
    contexts = unique_index(read_jsonl(q17 / JOURNALS[0]), CONTEXT_KEY)
    if set(contexts) != set(main["contexts"]):
        raise ValueError("Q17 contexts are incomplete. Finish Q17 before initializing Q8.")
    predictions = unique_index(validated_results(q17, dataset), PREDICTION_KEY)
    for reference, current in ((main["contexts"], contexts), (main["predictions"], predictions)):
        if any(current[key] != value for key, value in reference.items() if key[1] == "original"):
            raise ValueError("Q0 and Q17 must share the same Original records.")
    names = (*manifest["snapshot_hashes"], "manifest.json", "runtime_compressor.json", "runtime_judge.json", *JOURNALS)
    return main, {"run": q17, "hashes": {name: file_hash(q17 / name) for name in names}, "prompts": protocol["prompts"]}


def originals(dataset, contexts, predictions):
    keys = [doc["key"] for doc in dataset["documents"]]
    return ([contexts[(key, "original")] for key in keys],
            [predictions[(key, "original", hid)] for key in keys for hid in dataset["hypotheses"]])


def reference_location(path, output):
    try:
        return os.path.relpath(path, output)
    except ValueError:  # Windows references can be on a different drive.
        return str(path)


def initialize(main_path, q17_path, output):
    out = Path(output).resolve()
    if out.exists():
        raise ValueError("Q8 output already exists. Continue its stages or use a new output path.")
    main, q17 = inspect_references(main_path, q17_path)
    if any(out.is_relative_to(reference["run"]) for reference in (main, q17)):
        raise ValueError("Use a separate Q8 directory outside both reference runs.")
    dataset = main["dataset"]
    keys = main["manifest"]["document_keys"]
    assignment = balanced_assignment(keys, dataset["hypotheses"])
    prompts = prompt_spec(dataset, assignment, q17["prompts"])
    original_contexts, original_predictions = originals(dataset, main["contexts"], main["predictions"])
    references = {arm: {"path": reference_location(ref["run"], out), "hashes": ref["hashes"]}
                  for arm, ref in (("q0", main), ("q17", q17))}
    sources = source_hashes(main["memory"])
    identity = {"main": main["manifest"]["identity"],
                "references": {arm: digest(ref["hashes"]) for arm, ref in references.items()},
                "code": digest(sources), "assignment": digest(assignment), "prompts": digest(prompts),
                "memory_policy": digest(main["memory"])}
    marker = {"mode": EXPERIMENT, "protocol_identity": digest(identity)}
    protocol = {"identity": identity, "memory_policy": main["memory"], "question_context_policy": marker,
                "runtime": {role: {**details, "question_context_policy": marker}
                            for role, details in main["frozen"]["runtime"].items()}}
    snapshots = {"config.json": main["config"], "models.json": main["models"], "dataset.json": dataset,
                 "environment.json": read_json(main["run"] / "environment.json"), "prompts.json": prompts,
                 "prepared_manifest.json": read_json(main["run"] / "prepared_manifest.json"),
                 "assignment.json": assignment, "frozen_protocol.json": protocol,
                 "lineage.json": {"references": references, "original_contexts_sha256": digest(original_contexts),
                                  "original_predictions_sha256": digest(original_predictions)}}
    size, hypotheses = len(keys), len(dataset["hypotheses"])
    manifest = {"experiment": EXPERIMENT, "created_utc": now(), "split": "test", "identity": identity,
                "conditions": list(CONDITIONS), "document_keys": keys, "document_count": size,
                "hypotheses_per_document": hypotheses, "expected_contexts": size * len(CONDITIONS),
                "expected_judgments": size * hypotheses * len(CONDITIONS),
                "expected_new_compressions": size * (len(CONDITIONS) - 1),
                "expected_new_judgments": size * hypotheses * (len(CONDITIONS) - 1),
                "reused_original_judgments": len(original_predictions), "code_hashes": sources}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=f".{out.name}-") as temporary:
        staging = Path(temporary) / "run"
        for name, value in snapshots.items():
            write_json(staging / name, value)
        for relative in sources:
            target = staging / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        write_rows(staging / JOURNALS[0], original_contexts)
        write_rows(staging / JOURNALS[1], original_predictions)
        manifest["snapshot_hashes"] = {name: file_hash(staging / name) for name in snapshots}
        write_json(staging / "manifest.json", manifest)
        staging.rename(out)
    print(f"Initialized Q8: {size} documents; new compression={manifest['expected_new_compressions']}; "
          f"new judgment records={manifest['expected_new_judgments']}; Original reused={len(original_predictions)}.")
    print(f"Frozen assignment and exposure counts: {out / 'assignment.json'}")


def reference_paths(run):
    references = read_json(run / "lineage.json")["references"]
    paths = {arm: (run / ref["path"]).resolve() for arm, ref in references.items()}
    for arm, ref in references.items():
        for name, expected in ref["hashes"].items():
            path = paths[arm] / name
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"Reference {arm.upper()} is missing or changed: {path}")
    return paths


def load_subset_run(path, repair=False):
    run = Path(path).resolve()
    manifest = read_json(run / "manifest.json")
    if manifest.get("experiment") != EXPERIMENT:
        raise ValueError("Use the Q8 directory created by run_question_subset.py init.")
    for name, expected in manifest["snapshot_hashes"].items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Q8 snapshot changed: {name}")
    protocol = read_json(run / "frozen_protocol.json")
    memory = policy_spec()
    if (protocol["identity"] != manifest["identity"] or protocol["memory_policy"] != memory or
            manifest["code_hashes"] != source_hashes(memory)):
        raise ValueError("Q8 code or execution protocol changed. Continue with the source that initialized this run.")
    for relative, expected in manifest["code_hashes"].items():
        if file_hash(run / "source" / relative) != expected:
            raise ValueError(f"Archived Q8 source changed: {relative}")
    reference_paths(run)
    config, models, dataset = (read_json(run / name) for name in ("config.json", "models.json", "dataset.json"))
    keys, ids = [doc["key"] for doc in dataset["documents"]], list(dataset["hypotheses"])
    if read_json(run / "assignment.json") != balanced_assignment(keys, ids):
        raise ValueError("Q8 assignment does not match the frozen document and hypothesis IDs.")
    contexts = unique_index(read_jsonl(run / JOURNALS[0], repair=repair), CONTEXT_KEY)
    predictions = unique_index(read_jsonl(run / JOURNALS[1], repair=repair), PREDICTION_KEY)
    allowed = {(doc, condition) for doc in keys for condition in CONDITIONS}
    if set(contexts) - allowed or set(predictions) - {(*key, hid) for key in allowed for hid in ids}:
        raise ValueError("Unexpected Q8 context or judgment keys.")
    original_contexts, original_predictions = originals(dataset, contexts, predictions)
    lineage = read_json(run / "lineage.json")
    if (digest(original_contexts) != lineage["original_contexts_sha256"] or
            digest(original_predictions) != lineage["original_predictions_sha256"]):
        raise ValueError("Reused Original records changed.")
    memory_sha = runtime_policy_marker(memory)["policy_sha256"]
    for row in (*contexts.values(), *predictions.values()):
        skipped_judgment = "hypothesis" in row and row["status"] == "compression_failed"
        if row["condition"] == "original" or skipped_judgment:
            continue
        generation = row.get("generation") or {}
        if (generation.get("question_context_policy") != protocol["question_context_policy"] or
                generation.get("decoding_policy") != "explicit_greedy_v1" or
                generation.get("attention_memory_policy", {}).get("policy_sha256") != memory_sha):
            raise ValueError("Q8 output does not use the frozen question, decoding and memory policies.")
    return run, manifest, config, models, dataset
