import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


@pytest.mark.parametrize(
    "prefix,length,start,expected",
    [
        (0, 8, 10, ([None], [8])),
        (8, 9, 10, ([None, "adapter"], [2, 7])),
        (10, 7, 10, (["adapter"], [7])),
        (20, 1, 10, (["adapter"], [1])),
    ],
)
def test_token_segments_respect_lora_start_after_cache_hit(
    prefix, length, start, expected
):
    source = ROOT / "lora/lora_manager.py"
    tree = ast.parse(source.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "prepare_lora_batch"
    )
    # Exercise the actual selective-prefill branch without loading CUDA libraries.
    branch = ast.Module(body=method.body[:2], type_ignores=[])
    fn = ast.FunctionDef(
        name="route", args=method.args, body=branch.body, decorator_list=[]
    )
    ast.fix_missing_locations(fn)
    namespace = {"ForwardBatch": object}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace
    )
    manager = SimpleNamespace(prepare_lora_token_segments=Mock())
    batch = SimpleNamespace(
        lora_start_lens=[start],
        lora_ids=["adapter"],
        extend_prefix_lens_cpu=[prefix],
        extend_seq_lens_cpu=[length],
        forward_mode=SimpleNamespace(is_extend=lambda: True),
    )
    namespace["route"](manager, batch)
    manager.prepare_lora_token_segments.assert_called_once_with(
        lora_ids=expected[0], segment_lens=expected[1]
    )


def test_selective_request_shares_only_base_cache_namespace():
    import copy

    source = ROOT / "managers/schedule_batch.py"
    tree = ast.parse(source.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "__init__"
        and any(
            isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "lora_start" for t in x.targets)
            for x in n.body
        )
    )
    index = next(
        i
        for i, x in enumerate(method.body)
        if isinstance(x, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "lora_start" for t in x.targets)
    )
    params = SimpleNamespace(custom_params={"lora_start_len": 10})
    req = SimpleNamespace()
    ns = {
        "sampling_params": params,
        "self": req,
        "copy": copy,
        "lora_id": "adapter",
        "extra_key": None,
        "origin_input_ids": list(range(17)),
    }
    exec(
        compile(
            ast.Module(body=method.body[index : index + 2], type_ignores=[]),
            str(source),
            "exec",
        ),
        ns,
    )
    assert ns["extra_key"] is None
    assert params.custom_params == {"lora_start_len": 10}
    assert req.sampling_params.custom_params == {
        "lora_start_len": 10,
        "skip_radix_cache_insert": True,
        "cache_prefix_len": 10,
        "cache_original_prefix_len": 10,
    }
