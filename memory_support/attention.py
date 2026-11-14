"""Full-context SDPA evaluated one query head at a time during long prefills.

No token axis is split or shortened. GQA heads retain their original KV head.
The native math kernel still computes each complete attention distribution.
"""

from contextlib import contextmanager
from pathlib import Path

from contract_experiment.storage import digest, file_hash


def source_manifest():
    root = Path(__file__).resolve().parent.parent
    names = ["run_memory_safe.py", "memory_support/__init__.py",
             "memory_support/attention.py", "memory_support/runner.py"]
    return {name: file_hash(root / name) for name in names}


def policy_spec():
    return {"id": "headwise_math_prefill_v1", "query_threshold": 1024,
            "heads_per_call": 1, "sources": source_manifest(),
            "full_query_and_key_sequences": True, "dropout": 0.0,
            "chunked_kernel": "native SDPA MATH", "short_queries": "original native dispatcher"}


def headwise_sdpa(native, query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, *, scale=None, enable_gqa=False):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if dropout_p != 0.0:
        raise ValueError("The memory policy is inference-only and requires dropout_p=0.")
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("The memory policy requires [batch, heads, tokens, dim] tensors.")
    heads, kv_heads = query.shape[-3], key.shape[-3]
    if kv_heads != value.shape[-3] or heads < 1 or kv_heads < 1:
        raise ValueError("Invalid Q/K/V head counts.")
    if enable_gqa:
        if heads % kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        group_size = heads // kv_heads
    elif kv_heads in (1, heads):
        group_size = heads if kv_heads == 1 else 1
    else:
        raise ValueError("Different head counts require enable_gqa=True.")
    outputs = []
    # Selecting the math kernel explicitly avoids switching to a different fused
    # kernel merely because the head count has changed. Floating-point roundoff
    # can still differ; the CLI requires a recorded Dev output comparison.
    with sdpa_kernel(SDPBackend.MATH):
        for head in range(heads):
            kv_head = head // group_size
            mask = attn_mask
            if mask is not None and mask.ndim >= 3:
                if mask.shape[-3] == heads:
                    mask = mask[..., head:head + 1, :, :]
                elif mask.shape[-3] != 1:
                    raise ValueError("Unsupported attention-mask head dimension.")
            outputs.append(native(query[:, head:head + 1],
                key[:, kv_head:kv_head + 1], value[:, kv_head:kv_head + 1],
                attn_mask=mask, dropout_p=0.0, is_causal=is_causal,
                scale=scale, enable_gqa=False))
    return torch.cat(outputs, dim=1)


@contextmanager
def memory_attention(stats):
    import torch.nn.functional as functional

    native = functional.scaled_dot_product_attention

    def bounded(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, *, scale=None, enable_gqa=False):
        if query.shape[-2] <= 1024:
            return native(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        stats["chunked_calls"] += 1
        stats["max_query_tokens"] = max(stats["max_query_tokens"], query.shape[-2])
        stats["max_key_tokens"] = max(stats["max_key_tokens"], key.shape[-2])
        return headwise_sdpa(native, query, key, value, attn_mask, dropout_p,
                             is_causal, scale=scale, enable_gqa=enable_gqa)

    functional.scaled_dot_product_attention = bounded
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = native


def memory_model(base, spec):
    class MemorySafeModel(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print(f"Memory policy: {spec['id']} / long prefill / one head per call / full context", flush=True)

        def generate(self, messages):
            stats = {"chunked_calls": 0, "max_query_tokens": 0, "max_key_tokens": 0}
            with memory_attention(stats):
                generated = super().generate(messages)
            generated["attention_memory_policy"] = {
                "id": spec["id"], "policy_sha256": digest(spec), **stats}
            return generated

    return MemorySafeModel
