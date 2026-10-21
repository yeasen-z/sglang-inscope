# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import torch

from sglang.srt.feature_capture.graph import GraphFeatureCaptureRecorder
from sglang.srt.feature_capture.plan import SlotPlan, build_slot_plan
from sglang.srt.feature_capture.spec import validate


def spec(layers, positions, length):
    return validate(
        {"layers": layers, "positions": {"mode": "explicit", "index": positions}},
        num_layers=24,
        hidden_size=2,
        prompt_ids=list(range(length)),
    )


def forward(recorder, values):
    for layer in recorder.layers:
        recorder.capture_hook(layer=layer)(values + layer)


def test_mixed_requests_keep_native_offsets_and_hide_padding():
    recorder = GraphFeatureCaptureRecorder(layers=(8, 16, 24), capacity=8, device="cpu")
    plan = build_slot_plan(
        rids=("a", "ordinary", "b"),
        specs=(spec([8, 24], [0, -1], 3), None, spec([16, 24], [-1], 2)),
        extend_lens=(3, 2, 2),
        seq_lens=(3, 2, 2),
    )
    values = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    recorder.prepare(plan)
    forward(recorder, values)
    output = recorder.device_output()
    assert output.plan == plan
    expected = torch.cat([values[list(slot.rows)] + slot.layer for slot in plan])
    assert torch.equal(output.tensor, expected)
    assert output.tensor.shape == (6, 2)
    assert [slot.out_start for slot in output.plan] == [0, 2, 3]
    assert output.plan[-1].segments == (("a", (0, 2)), ("b", (1,)))


def test_replay_updates_indices_without_changing_buffer_addresses():
    recorder = GraphFeatureCaptureRecorder(layers=(8, 16), capacity=4, device="cpu")
    first = (SlotPlan(layer=8, rows=(3,), segments=(("a", (3,)),), out_start=0),)
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    recorder.prepare(first)
    forward(recorder, values)
    addresses = {
        layer: (
            recorder._indices[layer].data_ptr(),
            recorder._buffers[layer].data_ptr(),
        )
        for layer in recorder.layers
    }
    saved = recorder.device_output()
    expected_saved = saved.tensor.clone()
    second = (SlotPlan(layer=16, rows=(0, 2), segments=(("b", (0, 2)),), out_start=0),)
    recorder.prepare(second)
    forward(recorder, values + 100)
    output = recorder.device_output()
    assert torch.equal(output.tensor, values[[0, 2]] + 116)
    assert torch.equal(saved.tensor, expected_saved)
    assert saved.plan == first
    for layer, address in addresses.items():
        assert (
            recorder._indices[layer].data_ptr(),
            recorder._buffers[layer].data_ptr(),
        ) == address
    assert recorder._indices[8].tolist() == [0, 0, 0, 0]
    assert recorder._indices[16].tolist() == [0, 2, 0, 0]


def test_empty_plan_returns_no_output_after_prior_capture():
    recorder = GraphFeatureCaptureRecorder(layers=(8,), capacity=2, device="cpu")
    recorder.prepare(
        (SlotPlan(layer=8, rows=(1,), segments=(("a", (1,)),), out_start=0),)
    )
    forward(recorder, torch.ones(2, 2))
    assert recorder.device_output() is not None
    recorder.prepare(())
    forward(recorder, torch.ones(2, 2))
    assert recorder.device_output() is None
    assert recorder.capture_hook(layer=16) is None


@pytest.mark.parametrize(
    "slot,message",
    [
        (
            SlotPlan(layer=24, rows=(0,), segments=(("a", (0,)),), out_start=0),
            "does not include layer",
        ),
        (
            SlotPlan(
                layer=8, rows=(0, 1, 0), segments=(("a", (0, 1, 2)),), out_start=0
            ),
            "exceeds row capacity",
        ),
        (
            SlotPlan(layer=8, rows=(2,), segments=(("a", (0,)),), out_start=0),
            "outside token capacity",
        ),
        (
            SlotPlan(layer=8, rows=(-1,), segments=(("a", (0,)),), out_start=0),
            "outside token capacity",
        ),
        (
            SlotPlan(layer=8, rows=(0,), segments=(("a", (0,)),), out_start=1),
            "offsets must be contiguous",
        ),
        (
            SlotPlan(layer=8, rows=(0,), segments=(("a", (0, 1)),), out_start=0),
            "attribution does not match",
        ),
    ],
)
def test_invalid_plan_is_rejected_without_mutating_previous_plan(slot, message):
    recorder = GraphFeatureCaptureRecorder(layers=(8,), capacity=2, device="cpu")
    assert not recorder.can_prepare((slot,))
    with pytest.raises(ValueError, match=message):
        recorder.prepare((slot,))
    assert recorder.device_output() is None


def test_native_host_transport_attribution_survives_next_replay():
    recorder = GraphFeatureCaptureRecorder(layers=(8,), capacity=4, device="cpu")
    request_spec = spec([8], [0, -1], 3)
    plan = build_slot_plan(
        rids=("a",), specs=(request_spec,), extend_lens=(3,), seq_lens=(3,)
    )
    recorder.prepare(plan)
    values = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    forward(recorder, values)
    host = recorder.device_output().copy_to_host(lambda tensor: tensor)
    forward(recorder, values + 100)
    req = SimpleNamespace(
        rid="a",
        is_retracted=False,
        feature_capture=request_spec,
        feature_capture_values={},
        customized_info=None,
    )
    host.consume(SimpleNamespace(reqs=[req]), [None])
    assert torch.equal(req.feature_capture_values[(8, 0)], values[0] + 8)
    assert torch.equal(req.feature_capture_values[(8, 2)], values[2] + 8)
    (record,) = req.customized_info["features"]
    assert record["schema_version"] == "inscope.feature_capture.v1"
    assert record["positions"] == [0, 2]
    assert record["shape"] == [2, 2]


def test_chunked_plan_matches_native_eager_capture():
    from sglang.srt.feature_capture.recorder import FeatureCaptureRecorder

    plan = build_slot_plan(
        rids=("a", "b", "ordinary"),
        specs=(spec([8, 16], [0, 2, 4], 5), spec([16], [0, 3], 4), None),
        extend_lens=(2, 2, 3),
        seq_lens=(5, 4, 6),
        prefix_lens=(3, 2, 3),
    )
    graph = GraphFeatureCaptureRecorder(layers=(8, 16, 24), capacity=8, device="cpu")
    eager = FeatureCaptureRecorder(plan)
    graph.prepare(plan)
    values = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    forward(graph, values)
    for layer in (8, 16, 24):
        hook = eager.capture_hook(layer=layer)
        if hook is not None:
            hook(values + layer)
    assert graph.device_output().plan == eager.device_output().plan
    assert torch.equal(graph.device_output().tensor, eager.device_output().tensor)
    assert plan[0].rows == (1,)
    assert plan[1].rows == (1, 3)


def test_warmed_hook_does_not_allocate_tensors_or_dynamic_indices(monkeypatch):
    recorder = GraphFeatureCaptureRecorder(layers=(8,), capacity=2, device="cpu")
    recorder.prepare(
        (SlotPlan(layer=8, rows=(1,), segments=(("a", (1,)),), out_start=0),)
    )
    values = torch.ones(2, 2)
    forward(recorder, values)

    def forbidden(*args, **kwargs):
        raise AssertionError("hook tried allocating tensors after warmup")

    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch.Tensor, "new_empty", forbidden)
    forward(recorder, values)
