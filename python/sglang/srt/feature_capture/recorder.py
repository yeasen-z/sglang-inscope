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
"""Device recorder and scheduler-side transport for feature capture."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from sglang.srt.feature_capture.encoding import build_feature_records
from sglang.srt.feature_capture.plan import SlotPlan
from sglang.srt.managers.auxiliary_output import CommittedTokens


class FeatureCaptureRecorder:
    """Invocation-scoped recorder populated at model layer boundaries."""

    def __init__(self, plan: Sequence[SlotPlan]):
        self._plan = tuple(plan)
        self._slots = {slot.layer: slot for slot in self._plan}
        self._captured: set[int] = set()
        self._buffer: Optional[torch.Tensor] = None

    def capture_hook(self, *, layer: int):
        slot = self._slots.get(layer)
        if slot is None:
            return None

        def capture(value: torch.Tensor, *, owned: bool = False) -> None:
            self._capture(slot=slot, value=value)

        return capture

    def _capture(self, *, slot: SlotPlan, value: torch.Tensor) -> None:
        if slot.layer in self._captured:
            raise RuntimeError(f"feature capture layer {slot.layer} ran twice")
        if value.ndim != 2:
            raise RuntimeError(
                f"feature capture layer {slot.layer} expected a 2D hidden tensor, "
                f"got shape {tuple(value.shape)}"
            )
        if self._buffer is None:
            total_rows = sum(len(item.rows) for item in self._plan)
            self._buffer = value.new_empty((total_rows, value.shape[1]))
        elif self._buffer.shape[1] != value.shape[1]:
            raise RuntimeError("feature capture hidden size changed within one forward")

        rows = torch.tensor(slot.rows, dtype=torch.long, device=value.device)
        torch.index_select(
            value,
            0,
            rows,
            out=self._buffer[slot.out_start : slot.out_stop],
        )
        self._captured.add(slot.layer)

    def device_output(self) -> Optional[FeatureDeviceOutput]:
        if not self._plan:
            return None
        missing = set(self._slots) - self._captured
        if missing:
            raise RuntimeError(
                f"feature capture hooks did not run for layers {sorted(missing)}"
            )
        return FeatureDeviceOutput(tensor=self._buffer, plan=self._plan)


class FeatureDeviceOutput:
    def __init__(self, *, tensor: torch.Tensor, plan: tuple[SlotPlan, ...]):
        self.tensor = tensor
        self.plan = plan

    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> FeatureHostOutput:
        return FeatureHostOutput(tensor=copy_tensor(self.tensor), plan=self.plan)


class FeatureHostOutput:
    def __init__(self, *, tensor: torch.Tensor, plan: tuple[SlotPlan, ...]):
        self.tensor = tensor
        self.plan = plan

    def consume(self, batch, commits: Sequence[Optional[CommittedTokens]]) -> None:
        requests = {req.rid: req for req in batch.reqs}
        for slot in self.plan:
            row = slot.out_start
            for rid, positions in slot.segments:
                req = requests[rid]
                for position in positions:
                    if not req.is_retracted:
                        key = (slot.layer, position)
                        if key not in req.feature_capture_values:
                            req.feature_capture_values[key] = self.tensor[row].clone()
                    row += 1
        for req in batch.reqs:
            if req.is_retracted or req.feature_capture is None:
                continue
            if req.customized_info is None:
                req.customized_info = {}
            req.customized_info["features"] = build_feature_records(
                spec=req.feature_capture,
                values=req.feature_capture_values,
            )


def attach_feature_output(*, logits_output, recorder) -> None:
    """Attach completed recorder output without entering the sampler field."""
    if logits_output is not None and recorder is not None:
        logits_output.feature_device_output = recorder.device_output()
