"""Explicit recovery of verified greedy compressor outputs, never old judgments.

This migration supports only the archived implementation affected by D012.
It reads cached generation_config.json files on CPU; it never loads weights.
"""

import json
import os
from pathlib import Path

from . import CONDITIONS
from .prompts import compression_messages, select_units
from .runtime import (DECODING_POLICY, effective_greedy_config, greedy_overrides,
                      first_stop, load_tokenizer, render, token_count)
from .settings import environment, load_run, validate_config, validate_models
from .storage import (append_jsonl, digest, file_hash, now, read_json,
                      read_jsonl, run_lock, unique_index, write_json)


LEGACY_CODE_DIGEST = "e4593ad59b128165ee7bdbd5441d0d94a9c063c522cfba5156fed7e252f8ab1a"


def verified_source(path):
    source = Path(path)
    manifest = read_json(source / "manifest.json")
    hashes = manifest["code_hashes"]
    if digest(hashes) != LEGACY_CODE_DIGEST or manifest["identity"]["code"] != LEGACY_CODE_DIGEST:
        raise ValueError("Unsupported legacy source. Recovery only accepts the archived D012 implementation.")
    for relative, expected in hashes.items():
        if file_hash(source / "source" / relative) != expected:
            raise ValueError(f"Legacy source archive changed: {relative}")
    required = {"config.json", "models.json", "prepared_manifest.json", "prompts.json",
                "environment.json", "dataset.json"}
    if set(manifest["snapshot_hashes"]) != required:
        raise ValueError("Expected the original Dev snapshots, without a Test protocol.")
    for name, expected in manifest["snapshot_hashes"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"Legacy snapshot changed: {name}")
    config, models = read_json(source / "config.json"), read_json(source / "models.json")
    validate_config(config)
    validate_models(models, config)
    expected_identity = {"code": LEGACY_CODE_DIGEST,
        "config": digest(config), "models": digest(models),
        "prepared": digest(read_json(source / "prepared_manifest.json")),
        "prompts": digest(read_json(source / "prompts.json"))}
    if expected_identity != manifest["identity"]:
        raise ValueError("Legacy run identity does not match its snapshots.")
    return source, manifest, config, models, read_json(source / "dataset.json")


def verified_compressor_details(source, config, models):
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    details = read_json(source / "runtime_compressor.json")
    env = environment()
    if env["packages"]["transformers"] != "4.57.1":
        raise ValueError("Legacy decoding reconstruction requires transformers 4.57.1.")
    if any(details.get(key) != value for key, value in env.items()):
        raise ValueError("Legacy compressor environment differs from the current environment.")
    pinned = models["models"]["compressor"]
    if details["role"] != "compressor" or details["model_revision"] != pinned["revision"]:
        raise ValueError("Legacy compressor model does not match its lock.")
    defaults = GenerationConfig.from_pretrained(pinned["repo_id"], revision=pinned["revision"],
                                               local_files_only=True)
    # Bind only the real library's configuration method. No model is instantiated.
    class ConfigOnly(GenerationMixin):
        pass

    adapter = ConfigOnly()
    adapter.generation_config = defaults
    requested = GenerationConfig(**details["generation_config"])
    cap = config["generation"]["compressor_max_new_tokens"]
    old, _ = adapter._prepare_generation_config(requested, max_new_tokens=cap, logits_to_keep=1)
    fixed = effective_greedy_config(adapter, requested, cap)
    if old.do_sample or old.num_beams != 1 or old.to_dict() != fixed.to_dict():
        raise ValueError("Legacy compression decoding changed; compression outputs cannot be reused.")
    migrated = {**details, "load_seconds": 0.0, "decoding_policy": DECODING_POLICY,
                "generation_overrides": greedy_overrides(), "generation_config": fixed.to_dict()}
    return migrated, {"old_effective_do_sample": old.do_sample,
        "old_effective_num_beams": old.num_beams,
        "effective_config_identical": True, "effective_config_sha256": digest(fixed.to_dict())}


def validate_contexts(rows, dataset, config, models):
    """Validate every observed output, including failures and capped summaries."""
    import hashlib

    index = unique_index(rows, ("document", "condition"))
    docs = {doc["key"]: doc for doc in dataset["documents"]}
    expected = {(key, condition) for key in docs for condition in CONDITIONS}
    if set(index) != expected:
        raise ValueError("Recovery requires all five contexts for every source Dev document.")
    counter = load_tokenizer(config, models, "judge")
    compressor = load_tokenizer(config, models, "compressor")
    for key, row in index.items():
        doc = docs[key[0]]
        original_tokens = token_count(counter, doc["text"])
        if row["original_tokens"] != original_tokens:
            raise ValueError(f"Source token count changed: {key}")
        if key[1] == "original":
            if row["status"] != "ok" or row["text"] != doc["text"]:
                raise ValueError(f"Original context changed: {key}")
        else:
            generated = row["generation"]
            rendered = render(compressor, compression_messages(key[1], doc["units"]))
            if generated["rendered_prompt_sha256"] != hashlib.sha256(rendered.encode("utf-8")).hexdigest():
                raise ValueError(f"Compression prompt changed: {key}")
            if compressor.decode(generated["raw_token_ids"], skip_special_tokens=False) != generated["raw_text"]:
                raise ValueError(f"Raw compressor output changed: {key}")
            decoded = compressor.decode(generated["raw_token_ids"], skip_special_tokens=True)
            stop = first_stop(decoded, config["generation"]["stop_strings"])
            cleaned = (decoded[:stop[0]] if stop else decoded).strip()
            if cleaned != generated["text"]:
                raise ValueError(f"Cleaned compressor output changed: {key}")
            if key[1].startswith("extractive"):
                selected = select_units(generated["text"], doc["units"])
                if any(row[field] != selected[field] for field in ("status", "text", "selected_ids", "source_ranges", "error")):
                    raise ValueError(f"Extractive assembly changed: {key}")
            elif row["status"] != "ok" or row["text"] != generated["text"]:
                raise ValueError(f"Generative context changed: {key}")
        count = token_count(counter, row["text"]) if row["status"] == "ok" else None
        retention = count / original_tokens if count is not None else None
        if row["context_tokens"] != count or row["retention"] != retention:
            raise ValueError(f"Compression length changed: {key}")
    return index


def reuse_compression(source_path, target_path, check_only=False):
    source, old, config, models, dataset = verified_source(source_path)
    target, new, new_config, new_models, new_dataset = load_run(target_path)
    if source.resolve() == target.resolve():
        raise ValueError("Source and target must be separate directories.")
    with run_lock(target):
        if (old["split"] != "dev" or new["split"] != "dev" or
                old["document_count"] != old["full_split_count"] or
                new["document_count"] != new["full_split_count"] or
                dataset != new_dataset or config != new_config or models != new_models or
                {k: v for k, v in old["identity"].items() if k != "code"} !=
                {k: v for k, v in new["identity"].items() if k != "code"}):
            raise ValueError("Recovery requires matching full Dev data, prompts, models and settings.")
        if (read_json(source / "environment.json") != environment() or
                read_json(target / "environment.json") != environment()):
            raise ValueError("Recovery requires the same Python/platform/package versions.")
        if read_jsonl(target / "predictions.jsonl"):
            raise ValueError("Target has judgments. Use a fresh Dev run; old predictions are never imported.")
        source_hash = file_hash(source / "contexts.jsonl")
        rows = read_jsonl(source / "contexts.jsonl")
        index = validate_contexts(rows, dataset, config, models)
        originals = unique_index(read_jsonl(target / "contexts.jsonl"), ("document", "condition"))
        expected_originals = {key: row for key, row in index.items() if key[1] == "original"}
        if originals != expected_originals:
            raise ValueError("Target must contain only its unchanged initialized Original contexts.")
        details, proof = verified_compressor_details(source, config, models)
        provenance = {"created_utc": now(), "source_run": str(source.resolve()),
            "source_identity": old["identity"], "source_contexts_sha256": source_hash,
            "source_runtime_sha256": file_hash(source / "runtime_compressor.json"),
            "target_identity": new["identity"], "compressed_contexts": len(rows) - len(originals),
            "judgments_imported": 0, "verification": proof,
            "timing": "compression_seconds are original observed costs, not new execution time",
            "preserves_all_outputs_including_failures_and_caps": True}
        if file_hash(source / "contexts.jsonl") != source_hash:
            raise ValueError("Source journal changed during verification; stop its writer first.")
        if check_only:
            print(json.dumps(provenance, ensure_ascii=False, indent=2))
            print("Recovery check passed. No contexts or predictions were changed; no weights loaded.")
            return provenance
        # Provenance first: an interruption before replacement leaves Original-only
        # contexts and is safely repeatable. Journal replacement itself is atomic.
        write_json(target / "compression_reuse.json", provenance)
        write_json(target / "runtime_compressor.json", details)
        temp = target / "contexts.reuse.tmp"
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(target / "contexts.jsonl")
        append_jsonl(target / "events.jsonl", {"event": "compression_reused", "time": now(),
            "contexts": provenance["compressed_contexts"], "source_contexts_sha256": source_hash})
    print(f"Reused {provenance['compressed_contexts']} verified greedy compression outputs; imported 0 judgments.")
    return provenance
