"""Validated configuration and immutable run identity."""

import importlib.metadata
import platform
import re
import sys
import tomllib
from pathlib import Path

from . import CONDITIONS
from .prompts import prompt_spec
from .storage import digest, file_hash, read_json

ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config)
    return config


def validate_config(config):
    allowed = {"seed", "models", "runtime", "generation", "analysis"}
    if set(config) != allowed:
        raise ValueError(f"Config sections must be exactly {sorted(allowed)}; no retention target is supported.")
    expected = {
        "models": {"compressor", "judge"},
        "runtime": {"device", "quantization", "double_quant", "compute_dtype", "attention", "context_limit", "local_files_only"},
        "generation": {"compressor_max_new_tokens", "judge_max_new_tokens", "stop_strings"},
        "analysis": {"bootstrap_repetitions", "bootstrap_seed"},
    }
    for section, fields in expected.items():
        if set(config[section]) != fields:
            raise ValueError(f"Unexpected/missing settings in [{section}]; expected {sorted(fields)}")
    if config["models"] != {"compressor": "Qwen/Qwen2.5-7B-Instruct", "judge": "Qwen/Qwen3-8B"}:
        raise ValueError("This main protocol fixes Qwen2.5-7B-Instruct and Qwen3-8B.")
    runtime, generation = config["runtime"], config["generation"]
    if runtime["quantization"] != "nf4" or runtime["compute_dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("Use shared 4bit NF4 and an explicit compute dtype.")
    if not re.fullmatch(r"cuda:\d+", runtime["device"]) or runtime["attention"] not in {"sdpa", "eager"}:
        raise ValueError("Expected an explicit CUDA device and sdpa/eager attention.")
    for value in (runtime["context_limit"], generation["compressor_max_new_tokens"], generation["judge_max_new_tokens"],
                  config["analysis"]["bootstrap_repetitions"]):
        if type(value) is not int or value <= 0:
            raise ValueError("Context limits, safety caps and bootstrap repetitions must be positive integers.")
    if not isinstance(generation["stop_strings"], list) or any(not isinstance(s, str) or not s for s in generation["stop_strings"]):
        raise ValueError("stop_strings must be a list of nonempty strings.")
    if type(config["seed"]) is not int or type(config["analysis"]["bootstrap_seed"]) is not int:
        raise ValueError("Seeds must be integers.")
    if type(runtime["double_quant"]) is not bool or type(runtime["local_files_only"]) is not bool:
        raise ValueError("double_quant and local_files_only must be booleans.")


def code_hashes():
    paths = [ROOT / "run_experiment.py", ROOT / "requirements.txt", *sorted((ROOT / "contract_experiment").glob("*.py"))]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in paths}


def validate_models(lock, config):
    for role in ("compressor", "judge"):
        item = lock["models"][role]
        if item["repo_id"] != config["models"][role] or not re.fullmatch(r"[0-9a-f]{40}", item["revision"]):
            raise ValueError(f"Invalid or mismatched pinned model: {role}")


def identity(config, prepared_manifest, model_lock):
    return {"config": digest(config), "prepared": digest(prepared_manifest),
            "models": digest(model_lock), "code": digest(code_hashes()), "prompts": digest(prompt_spec())}


def environment():
    packages = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub", "tokenizers", "safetensors", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    installed = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "installed_packages": dict(sorted(installed.items()))}


def load_run(path):
    run = Path(path)
    manifest = read_json(run / "manifest.json")
    config = read_json(run / "config.json")
    validate_config(config)
    models = read_json(run / "models.json")
    prepared = read_json(run / "prepared_manifest.json")
    validate_models(models, config)
    if identity(config, prepared, models) != manifest["identity"]:
        raise ValueError("Run config, prompts, models or executable source changed. Use a new run; do not mix results.")
    for name, sha in manifest["snapshot_hashes"].items():
        if file_hash(run / name) != sha:
            raise ValueError(f"Run snapshot changed: {name}")
    dataset = read_json(run / "dataset.json")
    docs = dataset["documents"]
    if (manifest["document_keys"] != [doc["key"] for doc in docs] or
            manifest["conditions"] != list(CONDITIONS) or
            manifest["document_count"] != len(docs) or manifest["expected_judgments"] != len(docs) * 17 * 5):
        raise ValueError("Run manifest counts/keys do not match its dataset.")
    return run, manifest, config, models, dataset
