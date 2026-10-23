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
"""Encode host feature tensors into stable response records."""

from __future__ import annotations

try:
    import pybase64 as base64
except ImportError:
    import base64

import torch

from sglang.srt.feature_capture.spec import SCHEMA_VERSION, FeatureRequestSpec


def build_feature_records(
    *, spec: FeatureRequestSpec, values: dict[tuple[int, int], torch.Tensor]
) -> list[dict]:
    records = []
    for target in spec.targets:
        positions = [
            position
            for position in spec.positions
            if (target.layer, position) in values
        ]
        if not positions:
            continue
        tensor = torch.stack([values[(target.layer, pos)] for pos in positions])
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layer": target.layer,
                "site": target.site,
                "positions": positions,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "encoding": spec.encoding,
                "data": _encode_tensor(tensor=tensor, encoding=spec.encoding),
            }
        )
    return records


def _encode_tensor(*, tensor: torch.Tensor, encoding: str):
    tensor = tensor.detach().contiguous().cpu()
    if encoding == "float_list":
        return tensor.float().tolist()
    if encoding == "base64":
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return base64.b64encode(raw).decode("ascii")
    raise ValueError(f"unsupported feature encoding {encoding!r}")
