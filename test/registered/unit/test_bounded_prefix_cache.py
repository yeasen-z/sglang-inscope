"""Exercise the actual runtime's release path without loading CUDA libraries."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("skip_insert", [False, True])
def test_forks_release_resources_without_publishing_divergent_state(skip_insert):
    source = (
        Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache/common.py"
    )
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("release_kv_cache", "checkpoint_kv_cache", "_checkpoint_limit")
    ]
    namespace = {
        "Req": object,
        "BasePrefixCache": object,
        "_release_overallocated_kv_indices": Mock(),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    req = Mock()
    req.kv.holds_kv = True
    req.kv.is_kv_released = False
    req.kv.cache_protected_len = 2
    req.sampling_params = SimpleNamespace(custom_params={})
    req.kv.kv_allocated_len = 5
    req.owned_kv_len.return_value = 5
    batch_source = source.parents[1] / "managers/schedule_batch.py"
    assignment = next(
        node
        for node in ast.walk(ast.parse(batch_source.read_text()))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "skip_radix_cache_insert"
            for target in node.targets
        )
    )
    exec(
        compile(
            ast.Module(body=[assignment], type_ignores=[]), str(batch_source), "exec"
        ),
        {
            "self": req,
            "bootstrap_host": None,
            "FAKE_BOOTSTRAP_HOST": "fake",
            "positional_embed_overrides": None,
            "sampling_params": SimpleNamespace(
                custom_params={"skip_radix_cache_insert": skip_insert}
            ),
        },
    )
    assert req.skip_radix_cache_insert is skip_insert
    req.finished.return_value = True
    tree = Mock()
    tree.claim_kv_row.return_value = False
    namespace["release_kv_cache"](req, tree, checkpoint=True)
    assert tree.checkpoint.call_count == int(not skip_insert)
    tree.free_kv_row.assert_called_once_with(req.kv, [(2, 5)])
    tree.req_to_token_pool.free.assert_called_once_with(req)
    tree.on_release.assert_called_once_with(req, checkpointed=not skip_insert)
    req.kv.mark_kv_released.assert_called_once()
    req.finished.return_value = False
    tree.checkpoint.reset_mock()
    namespace["checkpoint_kv_cache"](req, tree)
    assert tree.checkpoint.call_count == int(not skip_insert)


@pytest.mark.parametrize("during_prefill", [False, True])
def test_bounded_fork_only_publishes_original_prefix_and_releases_private_tail(
    during_prefill,
):
    source = (
        Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache/common.py"
    )
    names = {"_checkpoint_limit", "checkpoint_kv_cache", "release_kv_cache"}
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = {
        "Req": object,
        "BasePrefixCache": object,
        "_release_overallocated_kv_indices": Mock(),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    req = Mock()
    req.skip_radix_cache_insert = True
    req.sampling_params = SimpleNamespace(
        custom_params={"cache_original_prefix_len": 3}
    )
    req.kv.holds_kv = True
    req.kv.is_kv_released = False
    req.kv.cache_protected_len = 2
    req.kv.kv_allocated_len = 7
    req.owned_kv_len.return_value = 7
    tree = Mock()
    tree.claim_kv_row.return_value = False
    tree.supports_mamba.return_value = False
    tree.supports_swa.return_value = False

    def checkpoint(r, up_to):
        r.kv.cache_protected_len = up_to

    tree.checkpoint.side_effect = checkpoint
    if during_prefill:
        req.finished.return_value = False
        req.extend_end = 5
        namespace["checkpoint_kv_cache"](req, tree)
        tree.checkpoint.assert_called_once_with(req, up_to=3)
        assert req.prefix_len == 5
    req.finished.return_value = True
    namespace["release_kv_cache"](req, tree, checkpoint=True)
    tree.checkpoint.assert_called_once_with(req, up_to=3)
    tree.free_kv_row.assert_called_once_with(req.kv, [(3, 7)])
    tree.req_to_token_pool.free.assert_called_once_with(req)
    tree.on_release.assert_called_once_with(req, checkpointed=not during_prefill)
    req.kv.mark_kv_released.assert_called_once()


@pytest.mark.parametrize("existing_limit, expected", [(None, 64), (32, 32), (96, 64)])
def test_fork_match_cap_does_not_trim_the_actual_request(existing_limit, expected):
    source = (
        Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache/common.py"
    )
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_req_radix_key"
    )
    namespace = {
        "Req": object,
        "BasePrefixCache": object,
        "RadixKey": SimpleNamespace,
        "array": list,
        "Optional": __import__("typing").Optional,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    tokens = list(range(100))
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={"cache_prefix_len": 64}),
        extra_key=None,
        cache_salt=None,
    )
    tree = SimpleNamespace(swa_reprefill_tail_tokens=lambda request: 0)
    key = namespace["_req_radix_key"](tree, req, tokens, existing_limit)
    assert key.limit == expected
    assert key.token_ids is tokens and len(tokens) == 100
