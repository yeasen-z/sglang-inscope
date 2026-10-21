# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Manual Qwen3 feature-capture E2E and Hugging Face parity check.

The test intentionally uses token ids for both the prompt and suffix. This
keeps chat-template boundary tokenization identical in SGLang and HF and
avoids treating a standalone-tokenized suffix as if it were a prompt suffix.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = "Explain why the sky appears blue in one sentence."
DEFAULT_LAYERS = (0, 1, 20, 35, 36)
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Qwen3 checkpoint (default: INSCOPE_QWEN3_MODEL)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--suffix", default="\nAnswer:")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Skip a multimodal checkpoint's encoder and serve its text tower",
    )
    parser.add_argument(
        "--repeat-input",
        type=int,
        default=1,
        help="Repeat the rendered token sequence to force long-prefill cases",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=-1,
        help="Engine chunked_prefill_size (-1 disables token chunking)",
    )
    parser.add_argument(
        "--enable-radix-cache",
        action="store_true",
        help="Enable radix cache for P3 cold/hot capture validation",
    )
    parser.add_argument(
        "--fp8-gemm-backend",
        choices=("auto", "triton"),
        default="auto",
        help="Use triton to avoid first-run DeepGEMM autotuning",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=1200,
        help="Allow first-run FP8 kernel tuning to finish (seconds)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--sglang-only",
        action="store_true",
        help="Run and optionally save SGLang captures",
    )
    modes.add_argument(
        "--hf-only",
        type=Path,
        metavar="CAPTURES_PT",
        help="Skip SGLang and compare a previously saved capture file with HF",
    )
    parser.add_argument(
        "--save-captures",
        type=Path,
        metavar="CAPTURES_PT",
        help="Atomically save SGLang tensors before starting the HF phase",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=0,
        help=(
            "Benchmark ordinary, one-layer, and requested-layer prefill after "
            "validation"
        ),
    )
    parser.add_argument(
        "--p3-multi-position",
        action="store_true",
        help=(
            "Validate one multi-position request against independent "
            "single-position requests"
        ),
    )
    parser.add_argument(
        "--p3-mixed-batch",
        action="store_true",
        help="Validate a mixed batch with no, single, and multi-target capture",
    )
    parser.add_argument(
        "--p3-radix-cache",
        action="store_true",
        help="Compare cold and prefix-cache-hit capture results",
    )
    parser.add_argument(
        "--p3-concurrent",
        action="store_true",
        help="Compare concurrent independent requests with serial baselines",
    )
    parser.add_argument(
        "--p3-ordinary-regression",
        action="store_true",
        help="Validate deterministic generation without a capture request",
    )
    parser.add_argument(
        "--p4-generate-features",
        action="store_true",
        help="Compare ordinary generation with prompt capture during generation",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="After shutting down SGLang, load the same checkpoint with HF and compare",
    )
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=2e-2)
    args = parser.parse_args()
    if args.benchmark_iterations < 0:
        parser.error("--benchmark-iterations must be non-negative")
    if args.repeat_input < 1:
        parser.error("--repeat-input must be positive")
    if args.hf_only and args.save_captures:
        parser.error("--save-captures cannot be combined with --hf-only")
    if args.sglang_only and args.compare_hf:
        parser.error("--sglang-only cannot be combined with --compare-hf")
    if not args.hf_only:
        args.model = args.model or os.getenv("INSCOPE_QWEN3_MODEL")
    if not args.model and not args.hf_only:
        parser.error("pass --model or set INSCOPE_QWEN3_MODEL")
    return args


def render_prompt(tokenizer, prompt: str, suffix: str) -> tuple[list[int], list[int]]:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        rendered + suffix,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    # Include a token that straddles the text/suffix boundary. Retokenizing the
    # suffix alone is not equivalent for BPE tokenizers and can make a genuine
    # terminal suffix look absent from the full prompt.
    suffix_start = len(rendered)
    first_suffix_token = next(
        (index for index, (_, end) in enumerate(offsets) if end > suffix_start),
        len(full_ids),
    )
    suffix_ids = full_ids[first_suffix_token:]
    if not suffix_ids:
        raise AssertionError("suffix must add at least one token")
    return full_ids, suffix_ids


def decode_record(record: dict[str, Any]) -> torch.Tensor:
    dtype_name = record["dtype"]
    if dtype_name not in DTYPES:
        raise AssertionError(f"unsupported response dtype {dtype_name!r}")
    if record["encoding"] == "float_list":
        tensor = torch.tensor(record["data"], dtype=DTYPES[dtype_name])
    elif record["encoding"] == "base64":
        raw = bytearray(base64.b64decode(record["data"], validate=True))
        tensor = torch.frombuffer(raw, dtype=DTYPES[dtype_name]).clone()
        tensor = tensor.reshape(record["shape"])
    else:
        raise AssertionError(f"unsupported response encoding {record['encoding']!r}")
    if list(tensor.shape) != record["shape"]:
        raise AssertionError(
            f"decoded shape {list(tensor.shape)} != advertised {record['shape']}"
        )
    return tensor


def validate_response_matrix(
    response: dict[str, Any], layers: list[int], positions: list[int]
) -> dict[int, torch.Tensor]:
    records = response.get("meta_info", {}).get("features")
    if not isinstance(records, list):
        raise AssertionError(f"response has no meta_info.features: {response.keys()}")
    returned_layers = [record["layer"] for record in records]
    if returned_layers != list(layers):
        raise AssertionError(
            f"returned layer order {returned_layers} != requested {list(layers)}"
        )
    by_layer = {record["layer"]: record for record in records}
    if set(by_layer) != set(layers):
        raise AssertionError(
            f"returned layers {sorted(by_layer)} != requested {sorted(layers)}"
        )

    decoded = {}
    for layer in layers:
        record = by_layer[layer]
        if record["schema_version"] != "inscope.feature_capture.v1":
            raise AssertionError(f"unexpected schema: {record['schema_version']}")
        if record["site"] != "block_output":
            raise AssertionError(f"unexpected site: {record['site']}")
        if record["positions"] != positions:
            raise AssertionError(
                f"layer {layer} positions {record['positions']} != {positions}"
            )
        value = decode_record(record)
        if value.shape[0] != len(positions):
            raise AssertionError(
                f"layer {layer} has {value.shape[0]} rows for {len(positions)} positions"
            )
        if not torch.isfinite(value).all():
            raise AssertionError(f"layer {layer} contains NaN or Inf")
        decoded[layer] = value

    if len(layers) > 1 and torch.equal(decoded[layers[0]], decoded[layers[-1]]):
        raise AssertionError(
            "first and last requested layers are unexpectedly identical"
        )
    return decoded


def validate_response(
    response: dict[str, Any], layers: list[int], position: int
) -> dict[int, torch.Tensor]:
    return {
        layer: value[0]
        for layer, value in validate_response_matrix(
            response, layers, [position]
        ).items()
    }


def validate_multi_position(engine, common, layers: list[int], input_len: int) -> None:
    requested = [0, input_len // 2, input_len - 1]
    response = engine.generate(
        **common,
        feature_capture={
            "layers": layers,
            "positions": {"mode": "explicit", "index": requested},
        },
    )
    combined = validate_response_matrix(response, layers, requested)

    for row, position in enumerate(requested):
        single_response = engine.generate(
            **common,
            feature_capture={
                "layers": layers,
                "positions": {"mode": "explicit", "index": [position]},
            },
        )
        single = validate_response(single_response, layers, position)
        for layer in layers:
            torch.testing.assert_close(combined[layer][row], single[layer])
    print(
        "P3 multi-position parity: PASS "
        f"layers={layers} positions={requested}"
    )


def validate_mixed_batch(engine, input_ids: list[int], layers: list[int]) -> None:
    batch_inputs = [input_ids[:-6], input_ids[:-3], input_ids]
    single_position = len(batch_inputs[1]) - 1
    multi_positions = [0, len(batch_inputs[2]) // 2, len(batch_inputs[2]) - 1]
    specs = [
        None,
        {
            "layers": [layers[0]],
            "positions": {"mode": "explicit", "index": [single_position]},
        },
        {
            "layers": layers,
            "positions": {"mode": "explicit", "index": multi_positions},
        },
    ]
    responses = engine.generate(
        input_ids=batch_inputs,
        sampling_params={"max_new_tokens": 0},
        feature_capture=specs,
    )
    if not isinstance(responses, list) or len(responses) != 3:
        raise AssertionError("mixed batch did not return three responses")
    ordinary_features = responses[0].get("meta_info", {}).get("features")
    if ordinary_features not in (None, []):
        raise AssertionError(
            "non-capture request unexpectedly returned non-empty features: "
            f"{ordinary_features!r}"
        )

    batch_single = validate_response(
        responses[1], [layers[0]], single_position
    )
    batch_multi = validate_response_matrix(responses[2], layers, multi_positions)

    independent_single_response = engine.generate(
        input_ids=batch_inputs[1],
        sampling_params={"max_new_tokens": 0},
        feature_capture=specs[1],
    )
    independent_single = validate_response(
        independent_single_response, [layers[0]], single_position
    )
    torch.testing.assert_close(
        batch_single[layers[0]], independent_single[layers[0]]
    )

    independent_multi_response = engine.generate(
        input_ids=batch_inputs[2],
        sampling_params={"max_new_tokens": 0},
        feature_capture=specs[2],
    )
    independent_multi = validate_response_matrix(
        independent_multi_response, layers, multi_positions
    )
    for layer in layers:
        torch.testing.assert_close(batch_multi[layer], independent_multi[layer])
    print(
        "P3 mixed-batch parity: PASS "
        f"lengths={[len(ids) for ids in batch_inputs]}"
    )


def validate_radix_cache(engine, common, layers: list[int], input_len: int) -> None:
    # Starting at a non-zero target proves that KV before the earliest target
    # remains reusable while every requested representation is recomputed.
    positions = [input_len // 2, input_len - 1]
    spec = {
        "layers": layers,
        "positions": {"mode": "explicit", "index": positions},
    }
    cold_response = engine.generate(**common, feature_capture=spec)
    cold = validate_response_matrix(cold_response, layers, positions)
    hot_response = engine.generate(**common, feature_capture=spec)
    cached_tokens = hot_response.get("meta_info", {}).get("cached_tokens", 0)
    if cached_tokens <= 0:
        raise AssertionError("second radix-cache request did not report a cache hit")
    hot = validate_response_matrix(hot_response, layers, positions)
    for layer in layers:
        torch.testing.assert_close(cold[layer], hot[layer])
    print(
        "P3 radix-cache parity: PASS "
        f"cached_tokens={cached_tokens} positions={positions}"
    )


def validate_concurrent(engine, input_ids: list[int], layers: list[int]) -> None:
    cases = [
        {
            "rid": "inscope-p3-concurrent-last",
            "input_ids": input_ids[:-6],
            "layers": [layers[0]],
            "positions": [len(input_ids[:-6]) - 1],
        },
        {
            "rid": "inscope-p3-concurrent-middle",
            "input_ids": input_ids[:-3],
            "layers": [layers[1], layers[2]],
            "positions": [0, len(input_ids[:-3]) // 2],
        },
        {
            "rid": "inscope-p3-concurrent-multi",
            "input_ids": input_ids,
            "layers": list(reversed(layers)),
            "positions": [1, len(input_ids) - 1],
        },
    ]

    def kwargs(case):
        return {
            "rid": case["rid"],
            "input_ids": case["input_ids"],
            "sampling_params": {"max_new_tokens": 0},
            "feature_capture": {
                "layers": case["layers"],
                "positions": {
                    "mode": "explicit",
                    "index": case["positions"],
                },
            },
        }

    async def submit_all():
        return await asyncio.gather(
            *(engine.async_generate(**kwargs(case)) for case in cases)
        )

    concurrent_responses = engine.loop.run_until_complete(submit_all())
    for case, response in zip(cases, concurrent_responses):
        if response.get("meta_info", {}).get("id") != case["rid"]:
            raise AssertionError(
                f"response rid does not match concurrent request {case['rid']}"
            )
        concurrent = validate_response_matrix(
            response, case["layers"], case["positions"]
        )
        serial_case = case | {"rid": case["rid"] + "-serial"}
        serial_response = engine.generate(**kwargs(serial_case))
        serial = validate_response_matrix(
            serial_response, case["layers"], case["positions"]
        )
        for layer in case["layers"]:
            torch.testing.assert_close(concurrent[layer], serial[layer])
    print("P3 concurrent request isolation: PASS requests=3")


def validate_ordinary_generation(engine, input_ids: list[int]) -> None:
    kwargs = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": 8,
            "temperature": 0,
        },
    }
    first = engine.generate(**kwargs)
    second = engine.generate(**kwargs)
    for response in (first, second):
        features = response.get("meta_info", {}).get("features")
        if features not in (None, []):
            raise AssertionError(
                f"ordinary request unexpectedly returned features: {features!r}"
            )
        if response.get("meta_info", {}).get("completion_tokens", 0) <= 0:
            raise AssertionError("ordinary request generated no tokens")
    if first.get("output_ids") != second.get("output_ids"):
        raise AssertionError("deterministic ordinary requests returned different tokens")
    if first.get("text") != second.get("text"):
        raise AssertionError("deterministic ordinary requests returned different text")
    print(
        "P3 ordinary generation regression: PASS "
        f"completion_tokens={first['meta_info']['completion_tokens']}"
    )


def validate_generate_with_features(
    engine, input_ids: list[int], layers: list[int]
) -> None:
    common = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": 8,
            "temperature": 0,
        },
    }
    ordinary = engine.generate(**common)
    captured = engine.generate(
        **common,
        feature_capture={
            "layers": layers,
            "positions": {"mode": "explicit", "index": [-1]},
        },
    )
    if ordinary.get("output_ids") != captured.get("output_ids"):
        raise AssertionError("generate+features changed generated token ids")
    if ordinary.get("text") != captured.get("text"):
        raise AssertionError("generate+features changed generated text")
    validate_response(captured, layers, len(input_ids) - 1)
    if captured.get("meta_info", {}).get("completion_tokens") != 8:
        raise AssertionError("generate+features did not complete normal decoding")
    print(
        "P4 generate+features parity: PASS "
        f"completion_tokens={captured['meta_info']['completion_tokens']}"
    )


def percentile_ms(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))
    return ordered[index] * 1000


def benchmark_prefill(engine, common, layers: list[int], iterations: int) -> None:
    cases = {
        "ordinary": None,
        "capture_1_layer": {"layers": [layers[0]]},
        "capture_all_layers": {"layers": layers},
    }
    print(f"benchmark iterations={iterations} (one warmup per case)")
    for name, capture in cases.items():
        engine.generate(**common, feature_capture=capture)
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            engine.generate(**common, feature_capture=capture)
            samples.append(time.perf_counter() - started)
        print(
            f"timing {name}: mean={statistics.fmean(samples) * 1000:.3f}ms "
            f"p50={percentile_ms(samples, 0.50):.3f}ms "
            f"p95={percentile_ms(samples, 0.95):.3f}ms"
        )


def run_sglang(args, input_ids: list[int], suffix_ids: list[int]):
    # Read by the parent and inherited by scheduler subprocesses at startup.
    os.environ["SGLANG_ENABLE_FEATURE_CAPTURE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import sglang as sgl

    started = time.perf_counter()
    engine = sgl.Engine(
        model_path=args.model,
        tp_size=1,
        disable_radix_cache=not args.enable_radix_cache,
        disable_cuda_graph=True,
        chunked_prefill_size=args.chunked_prefill_size,
        mem_fraction_static=args.mem_fraction_static,
        watchdog_timeout=args.watchdog_timeout,
        fp8_gemm_runner_backend=args.fp8_gemm_backend,
        language_model_only=args.language_model_only,
    )
    startup_seconds = time.perf_counter() - started
    print(f"timing engine_startup={startup_seconds:.3f}s")
    try:
        common = {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 0},
        }
        if args.p3_radix_cache:
            validate_radix_cache(
                engine, common, list(args.layers), len(input_ids)
            )
        started = time.perf_counter()
        explicit = engine.generate(
            **common,
            feature_capture={
                "layers": args.layers,
                "positions": {"mode": "explicit", "index": [-1]},
            },
        )
        explicit_seconds = time.perf_counter() - started
        explicit_values = validate_response(explicit, args.layers, len(input_ids) - 1)

        started = time.perf_counter()
        suffix = engine.generate(
            **common,
            feature_capture={
                "layers": args.layers,
                "positions": {
                    "mode": "suffix_last",
                    "suffix_token_ids": suffix_ids,
                },
            },
        )
        suffix_seconds = time.perf_counter() - started
        suffix_values = validate_response(suffix, args.layers, len(input_ids) - 1)
        for layer in args.layers:
            torch.testing.assert_close(explicit_values[layer], suffix_values[layer])
        print(
            f"timing explicit_capture={explicit_seconds * 1000:.3f}ms "
            f"suffix_capture={suffix_seconds * 1000:.3f}ms"
        )
        if args.benchmark_iterations:
            benchmark_prefill(
                engine, common, list(args.layers), args.benchmark_iterations
            )
        if args.p3_multi_position:
            validate_multi_position(
                engine, common, list(args.layers), len(input_ids)
            )
        if args.p3_mixed_batch:
            validate_mixed_batch(engine, input_ids, list(args.layers))
        if args.p3_concurrent:
            validate_concurrent(engine, input_ids, list(args.layers))
        if args.p3_ordinary_regression:
            validate_ordinary_generation(engine, input_ids)
        if args.p4_generate_features:
            validate_generate_with_features(engine, input_ids, list(args.layers))
        return explicit_values
    finally:
        engine.shutdown()


def save_captures(
    path: Path,
    *,
    model: str,
    input_ids: list[int],
    suffix_ids: list[int],
    values: dict[int, torch.Tensor],
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "format": "inscope.qwen3_capture.v1",
        "model": model,
        "input_ids": input_ids,
        "suffix_ids": suffix_ids,
        "layers": list(values),
        "values": {layer: value.cpu() for layer, value in values.items()},
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f"saved captures: {path}")


def load_captures(path: Path):
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=True)
    if payload.get("format") != "inscope.qwen3_capture.v1":
        raise ValueError(f"unsupported capture file format in {path}")
    required = {"model", "input_ids", "suffix_ids", "layers", "values"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"capture file is missing: {sorted(missing)}")
    if set(payload["layers"]) != set(payload["values"]):
        raise ValueError("capture file layers do not match stored tensors")
    return payload


def compare_hf(args, input_ids: list[int], actual: dict[int, torch.Tensor]) -> None:
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()
    print(f"timing hf_model_load={time.perf_counter() - started:.3f}s")
    last_block_output = []

    def capture_last_block(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        last_block_output.append(value.detach())

    hook = model.model.layers[-1].register_forward_hook(capture_last_block)
    ids = torch.tensor([input_ids], device="cuda")
    try:
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        print(f"timing hf_forward={(time.perf_counter() - started) * 1000:.3f}ms")
    finally:
        hook.remove()

    failures = []
    for layer in args.layers:
        # Transformers' final hidden_states entry is post-final-norm. InScope
        # layer N deliberately means the last decoder block output before that
        # norm, so obtain it from the block hook instead.
        source = (
            last_block_output[0]
            if layer == model.config.num_hidden_layers
            else outputs.hidden_states[layer]
        )
        expected = source[0, -1].float().cpu()
        observed = actual[layer].float().cpu()
        cosine = torch.nn.functional.cosine_similarity(observed, expected, dim=0).item()
        relative_l2 = (
            torch.linalg.vector_norm(observed - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1e-12)
        ).item()
        print(f"layer={layer:>3} cosine={cosine:.8f} relative_l2={relative_l2:.8g}")
        if cosine < args.min_cosine or relative_l2 > args.max_relative_l2:
            failures.append((layer, cosine, relative_l2))
    if failures:
        raise AssertionError(f"HF parity thresholds failed: {failures}")


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.hf_only:
        payload = load_captures(args.hf_only)
        args.model = args.model or payload["model"]
        args.layers = payload["layers"]
        print(
            f"loaded captures={args.hf_only} model={args.model} "
            f"prompt_tokens={len(payload['input_ids'])} layers={args.layers}"
        )
        compare_hf(args, payload["input_ids"], payload["values"])
        print("SGLang/Hugging Face hidden-state parity: PASS")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids, suffix_ids = render_prompt(tokenizer, args.prompt, args.suffix)
    input_ids = input_ids * args.repeat_input
    print(
        f"model={args.model} prompt_tokens={len(input_ids)} "
        f"suffix_tokens={len(suffix_ids)} layers={args.layers}"
    )
    captured = run_sglang(args, input_ids, suffix_ids)
    print("SGLang explicit/suffix capture parity: PASS")
    if args.save_captures:
        save_captures(
            args.save_captures,
            model=args.model,
            input_ids=input_ids,
            suffix_ids=suffix_ids,
            values=captured,
        )
    if args.compare_hf and not args.sglang_only:
        compare_hf(args, input_ids, captured)
        print("SGLang/Hugging Face hidden-state parity: PASS")


if __name__ == "__main__":
    main()
