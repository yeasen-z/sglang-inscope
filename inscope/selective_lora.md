# Selective LoRA branches over base-model context

Use `sampling_params.custom_params.lora_start_len` to activate an adapter at an
absolute token boundary. Tokens before the boundary use the base model; the
boundary token and subsequent tokens use the adapter. Decode tokens use the
adapter normally. Omitting the field, or setting it to zero, preserves ordinary
whole-request LoRA behavior.

For a safety footnote, concatenate the original context token IDs with the token
IDs of `<|im_end|>\nAre there potential risks?\n`, and set `lora_start_len` to the
number of original context tokens. The appended `<|im_end|>` is already adapted.

```python
payload = {
    "input_ids": context_ids + question_ids,
    "lora_path": "safety",
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 1024,
        "custom_params": {"lora_start_len": len(context_ids)},
    },
}
```

Use a PEFT adapter targeting only `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj`. Embedding and `lm_head` adapters are not
covered by this token-segment path. Start the server with:

```bash
--enable-lora --lora-paths safety=/path/to/adapter \
--lora-backend triton --max-loras-per-batch 2 --disable-cuda-graph
```

The two LoRA slots cover the base model and the adapter. The validated setup is
TP1, one adapter, full-attention KV, and eager routing. Mamba and sliding-window
cache publication are not supported by this feature.

## KV semantics

Selective requests share the base-model radix namespace. The runtime caps cache
matching at `lora_start_len` and publishes only the computed original-context
prefix, including during chunked prefill. The adapted question and generated
continuation remain private and are released when the request finishes. A
subsequent ordinary base request can therefore reuse the prefix without reading
adapted KV. Caller-provided cache classification and salt remain in effect.
Ordinary non-selective LoRA requests retain their adapter-specific namespace.

This uses the existing eager token-segment routing backend; no per-layer Python
masking or prefix recomputation is added.

## Validation

CPU routing and cache-lifecycle tests:

```bash
python -m pytest -q test/registered/unit/test_selective_lora.py \
    test/registered/unit/test_bounded_prefix_cache.py
```

A Qwen3-8B-FP8 live canary exercised a 61,318-token context through cold selective
prefill, an ordinary base request, then base-seeded selective prefill. The base
request matched exactly the original 61,318 tokens, and the cold/warm first-label
log probabilities differed by at most 7.63e-6. A separate 2,048-context greedy
readout completed with valid format and stop tokens for every response.
