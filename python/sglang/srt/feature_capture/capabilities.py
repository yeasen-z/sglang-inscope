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
"""Fail-fast capability checks for hidden-feature capture.

The v1 row plan assumes one complete, text-only token axis. Configurations
that split that axis or introduce additional forwards must be rejected rather
than returning plausible vectors with the wrong attribution.
"""

from __future__ import annotations

import msgspec

from sglang.srt.environ import envs

SUPPORTED_SITE = "block_output"


class FeatureCaptureCapabilityError(ValueError):
    """Feature capture was enabled for an unverified runtime configuration."""


class FeatureCaptureCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """Startup facts needed by the v1 feature-capture capability gate."""

    device: str
    model_name: str
    model_sites: tuple[str, ...] = ()
    pp_size: int = 1
    dp_attention: bool = False
    attn_cp_size: int = 1
    dcp_size: int = 1
    two_batch_overlap: bool = False
    attn_tp_input_scattered: bool = False
    layernorm_sp: bool = False
    speculative: bool = False
    disaggregation_mode: str = "null"
    multimodal: bool = False
    split_prefill: bool = False


def is_enabled() -> bool:
    """Return the process-static opt-in switch."""
    return envs.SGLANG_ENABLE_FEATURE_CAPTURE.get()


def validate_capabilities(capabilities: FeatureCaptureCapabilities) -> None:
    """Reject every runtime shape whose capture semantics are unverified."""
    reasons = _unsupported_reasons(capabilities)
    if reasons:
        raise FeatureCaptureCapabilityError(
            "feature capture is not supported by this runtime: " + "; ".join(reasons)
        )


def _unsupported_reasons(capabilities: FeatureCaptureCapabilities) -> tuple[str, ...]:
    reasons: list[str] = []
    if capabilities.device != "cuda":
        reasons.append(f"device={capabilities.device!r} (CUDA is required)")
    if capabilities.pp_size != 1:
        reasons.append(f"pp_size={capabilities.pp_size} (pipeline parallelism)")
    if capabilities.dp_attention:
        reasons.append("DP attention is enabled")
    if capabilities.attn_cp_size != 1:
        reasons.append(f"attn_cp_size={capabilities.attn_cp_size}")
    if capabilities.dcp_size != 1:
        reasons.append(f"dcp_size={capabilities.dcp_size}")
    if capabilities.two_batch_overlap:
        reasons.append("two-batch overlap is enabled")
    if capabilities.attn_tp_input_scattered:
        reasons.append("attention TP input scattering is enabled")
    if capabilities.layernorm_sp:
        reasons.append("layernorm sequence parallelism is enabled")
    if capabilities.speculative:
        reasons.append("speculative decoding is enabled")
    if capabilities.disaggregation_mode != "null":
        reasons.append(
            f"disaggregation_mode={capabilities.disaggregation_mode!r}"
        )
    if capabilities.multimodal:
        reasons.append("multimodal input is enabled")
    if capabilities.split_prefill:
        reasons.append("split prefill is enabled")
    if SUPPORTED_SITE not in capabilities.model_sites:
        reasons.append(
            f"{capabilities.model_name} has no verified {SUPPORTED_SITE!r} adapter"
        )
    return tuple(reasons)
