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
"""Stable device buffers for native feature capture inside CUDA graphs."""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.feature_capture.plan import SlotPlan
from sglang.srt.feature_capture.recorder import FeatureDeviceOutput


class GraphFeatureCaptureRecorder:
    """Capture fixed-size rows, then publish only the current native slot plan.

    ``prepare`` runs before replay, on the replay stream. Model warmup creates
    each layer's output buffer once. Subsequent hooks only perform index_select
    with stable input/output addresses, including layers absent from a request.
    ``device_output`` runs after replay and owns its compact result so a later
    replay cannot overwrite an asynchronous copy to the scheduler.
    """

    def __init__(self, *, layers: Sequence[int], capacity: int, device):
        self.layers = tuple(layers)
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("graph capture capacity must be positive")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("graph capture requires unique, non-empty layers")
        self._indices = {
            layer: torch.zeros(self.capacity, dtype=torch.long, device=device)
            for layer in self.layers
        }
        self._buffers: dict[int, torch.Tensor] = {}
        self._plan: tuple[SlotPlan, ...] = ()

    def _validate_plan(self, plan: Sequence[SlotPlan]) -> None:
        seen = set()
        output_offset = 0
        for slot in plan:
            if slot.layer not in self._indices:
                raise ValueError(f"graph capture does not include layer {slot.layer}")
            if slot.layer in seen:
                raise ValueError(f"duplicate graph capture slot for layer {slot.layer}")
            seen.add(slot.layer)
            if len(slot.rows) > self.capacity:
                raise ValueError("graph capture slot exceeds row capacity")
            if any(row < 0 or row >= self.capacity for row in slot.rows):
                raise ValueError("graph capture source row is outside token capacity")
            if slot.out_start != output_offset:
                raise ValueError("graph capture output offsets must be contiguous")
            if sum(len(positions) for _, positions in slot.segments) != len(slot.rows):
                raise ValueError(
                    "graph capture slot attribution does not match its rows"
                )
            output_offset = slot.out_stop

    def can_prepare(self, plan: Sequence[SlotPlan]) -> bool:
        try:
            self._validate_plan(plan)
        except ValueError:
            return False
        return True

    def prepare(self, plan: Sequence[SlotPlan]) -> None:
        """Update stable index buffers outside graph capture/replay."""
        plan = tuple(plan)
        self._validate_plan(plan)
        for indices in self._indices.values():
            indices.zero_()
        for slot in plan:
            indices = self._indices[slot.layer]
            indices[: len(slot.rows)].copy_(
                torch.tensor(slot.rows, dtype=torch.long, device=indices.device)
            )
        self._plan = plan

    def capture_hook(self, *, layer: int):
        if layer not in self._indices:
            return None

        def capture(value: torch.Tensor, *, owned: bool = False) -> None:
            self._capture(layer=layer, value=value)

        return capture

    def _capture(self, *, layer: int, value: torch.Tensor) -> None:
        if value.ndim != 2:
            raise RuntimeError("graph feature capture requires a 2D hidden tensor")
        buffer = self._buffers.get(layer)
        if buffer is None:
            buffer = value.new_empty((self.capacity, value.shape[1]))
            self._buffers[layer] = buffer
        elif buffer.shape[1] != value.shape[1] or buffer.dtype != value.dtype:
            raise RuntimeError("graph feature capture hidden shape or dtype changed")
        torch.index_select(value, 0, self._indices[layer], out=buffer)

    def device_output(self) -> FeatureDeviceOutput | None:
        """Compact real rows outside the graph without exposing replay storage."""
        if not self._plan:
            return None
        missing = [slot.layer for slot in self._plan if slot.layer not in self._buffers]
        if missing:
            raise RuntimeError(f"graph capture hooks did not warm up layers {missing}")
        tensor = torch.cat(
            [self._buffers[slot.layer][: len(slot.rows)] for slot in self._plan], dim=0
        )
        return FeatureDeviceOutput(tensor=tensor, plan=self._plan)
