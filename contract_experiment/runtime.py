"""Lazy Hugging Face adapter. Importing core modules never loads torch or models."""

import gc
import hashlib
import os
import random
import re
import time
from pathlib import Path

from .settings import environment, validate_models
from .storage import now, read_json, write_json


def lock_models(config, output):
    from huggingface_hub import HfApi
    path = Path(output)
    if path.exists():
        validate_models(read_json(path), config)
        print(f"Using existing immutable model lock: {path}")
        return
    api = HfApi()
    models = {role: {"repo_id": repo_id, "revision": api.model_info(repo_id).sha}
              for role, repo_id in config["models"].items()}
    write_json(path, {"created_utc": now(), "models": models})
    print(f"Pinned model revisions saved to {path}; no model weights downloaded.")


def download_models(config, lock):
    from huggingface_hub import snapshot_download
    validate_models(lock, config)
    for role, model in lock["models"].items():
        print(f"Downloading {role}: {model['repo_id']} @ {model['revision']}", flush=True)
        snapshot_download(model["repo_id"], revision=model["revision"],
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])


def load_tokenizer(config, lock, role):
    from transformers import AutoTokenizer
    model = lock["models"][role]
    tokenizer = AutoTokenizer.from_pretrained(model["repo_id"], revision=model["revision"],
        trust_remote_code=False, local_files_only=config["runtime"]["local_files_only"])
    if not tokenizer.chat_template:
        raise ValueError(f"Missing chat template: {role}")
    return tokenizer


def render(tokenizer, messages, judge=False):
    kwargs = {"enable_thinking": False} if judge else {}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def first_stop(text, stops):
    matches = []
    for stop in stops:
        position = text.find(stop)
        if stop.startswith("\n") and text.startswith(stop[1:]):
            position = 0  # The first generated line is also a line boundary.
        if position >= 0:
            matches.append((position, stop))
    return min(matches) if matches else None


DECODING_POLICY = "explicit_greedy_v1"


def greedy_overrides():
    # In transformers 4.57, fields equal to GenerationConfig's global defaults
    # can otherwise be replaced by model defaults (Qwen3 has do_sample=True).
    return {"use_model_defaults": False, "do_sample": False,
            "num_beams": 1, "use_cache": True}


def effective_greedy_config(model, requested, cap):
    effective, _ = model._prepare_generation_config(
        requested, **greedy_overrides(), max_new_tokens=cap, logits_to_keep=1)
    if effective.do_sample or effective.num_beams != 1:
        raise ValueError("Effective decoding is not greedy; no inference was started.")
    return effective


class LocalModel:
    def __init__(self, config, lock, role):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

        self.torch, self.config, self.role = torch, config, role
        runtime = config["runtime"]
        self.device = runtime["device"]
        if not torch.cuda.is_available():
            raise ValueError("CUDA-enabled PyTorch/NVIDIA GPU required. See README installation commands.")
        device_index = int(self.device.split(":")[1])
        torch.cuda.set_device(device_index)
        if runtime["compute_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bf16. Set compute_dtype=float16 before new Dev runs and use it for both models.")
        random.seed(config["seed"])
        np.random.seed(config["seed"] % (2 ** 32))
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.tokenizer = load_tokenizer(config, lock, role)
        model = lock["models"][role]
        dtype = getattr(torch, runtime["compute_dtype"])
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=runtime["double_quant"], bnb_4bit_compute_dtype=dtype)
        start = time.perf_counter()
        print(f"Loading {role}: shared 4bit NF4 / {runtime['compute_dtype']} / {self.device}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model["repo_id"], revision=model["revision"], quantization_config=quant,
            torch_dtype=dtype, device_map={"": self.device}, attn_implementation=runtime["attention"],
            trust_remote_code=False, local_files_only=runtime["local_files_only"], use_safetensors=True)
        self.model.eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.context_limit = min(runtime["context_limit"], self.model.config.max_position_embeddings)
        end_ids = self.model.generation_config.eos_token_id
        self.eos_ids = set(end_ids if isinstance(end_ids, list) else [end_ids])
        self.eos_ids.add(self.tokenizer.eos_token_id)
        self.eos_ids.discard(None)
        self.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=sorted(self.eos_ids), pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        cap = config["generation"][f"{role}_max_new_tokens"]
        self.generation_config = effective_greedy_config(self.model, self.generation_config, cap)
        print(f"Decoding verified: {role} / do_sample=False / num_beams=1 / model defaults disabled", flush=True)
        self.details = {**environment(), "role": role, "gpu": torch.cuda.get_device_name(),
            "cuda_version": torch.version.cuda, "device": self.device,
            "model_revision": model["revision"], "load_seconds": self.load_seconds,
            "quantization": quant.to_dict(), "attention": runtime["attention"],
            "decoding_policy": DECODING_POLICY, "generation_overrides": greedy_overrides(),
            "generation_config": self.generation_config.to_dict(),
            "chat_template": self.tokenizer.chat_template, "eos_token_ids": sorted(self.eos_ids),
            "context_limit": self.context_limit, "deterministic_algorithms": "enabled, warn_only=True"}

    def generate(self, messages):
        from transformers import StoppingCriteria, StoppingCriteriaList

        tokenizer, torch = self.tokenizer, self.torch
        cap = self.config["generation"][f"{self.role}_max_new_tokens"]
        stops = self.config["generation"]["stop_strings"]
        rendered = render(tokenizer, messages, judge=self.role == "judge")
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(self.device)
        input_tokens = inputs["input_ids"].shape[-1]
        if input_tokens + cap > self.context_limit:
            raise ValueError(f"Input ({input_tokens}) + safety cap ({cap}) exceeds context limit ({self.context_limit}). No input was truncated.")

        class StopOnDialogue(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                generated = tokenizer.decode(input_ids[0, input_tokens:], skip_special_tokens=True)
                return first_stop(generated, stops) is not None

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, generation_config=self.generation_config, max_new_tokens=cap,
                **greedy_overrides(),
                stopping_criteria=StoppingCriteriaList([StopOnDialogue()]),
                # Avoid allocating a full [prompt_length, vocabulary_size] logits tensor.
                logits_to_keep=1,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        ids = output[0, input_tokens:].tolist()
        raw_text = tokenizer.decode(ids, skip_special_tokens=False)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        occurrence = first_stop(text, stops)
        if occurrence is not None:
            position, matched = occurrence
            clean, reason = text[:position].strip(), "stop_string"
        else:
            clean, matched = text.strip(), None
            reason = "eos" if ids and ids[-1] in self.eos_ids else "max_tokens"
        return {
            "text": clean, "raw_text": raw_text, "raw_token_ids": ids,
            "decoding_policy": DECODING_POLICY,
            "stop_reason": reason, "matched_stop_string": matched,
            "cap_reached": len(ids) >= cap, "max_new_tokens": cap,
            "dialogue_flag": bool(re.search(r"(?:^|\n)(?:Human|User|Assistant):", text)),
            "input_tokens": input_tokens, "generated_tokens": len(ids), "generation_seconds": seconds,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        }

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
