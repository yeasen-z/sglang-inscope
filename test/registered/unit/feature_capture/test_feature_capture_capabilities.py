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
"""Capability-gate tests for request-scoped hidden-feature capture."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.environ import envs
from sglang.srt.feature_capture.capabilities import (
    FeatureCaptureCapabilities,
    FeatureCaptureCapabilityError,
    is_enabled,
    validate_capabilities,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


def _supported(**changes):
    values = {
        "device": "cuda",
        "model_name": "AdaptedModel",
        "model_sites": ("block_output",),
    }
    values.update(changes)
    return FeatureCaptureCapabilities(**values)


class TestFeatureCaptureCapabilityGate(CustomTestCase):
    def test_qwen35_text_tower_is_an_allowed_language_model_only_architecture(self):
        self.assertIn(
            "Qwen3_5ForConditionalGeneration",
            ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES,
        )

    def test_supported_minimal_runtime_passes(self):
        validate_capabilities(_supported())

    def test_each_unverified_runtime_shape_is_rejected(self):
        # Each input breaks an independent row-attribution or capture-site
        # invariant. Keeping this table complete prevents a newly wired caller
        # from silently omitting one startup check.
        unsupported = {
            "non_cuda": {"device": "cpu"},
            "pipeline_parallel": {"pp_size": 2},
            "dp_attention": {"dp_attention": True},
            "attention_context_parallel": {"attn_cp_size": 2},
            "decode_context_parallel": {"dcp_size": 2},
            "two_batch_overlap": {"two_batch_overlap": True},
            "scattered_tp_input": {"attn_tp_input_scattered": True},
            "layernorm_sequence_parallel": {"layernorm_sp": True},
            "speculative_decoding": {"speculative": True},
            "disaggregation": {"disaggregation_mode": "prefill"},
            "multimodal": {"multimodal": True},
            "split_prefill": {"split_prefill": True},
            "model_without_adapter": {"model_sites": ()},
        }
        for name, changes in unsupported.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                FeatureCaptureCapabilityError, "feature capture is not supported"
            ):
                validate_capabilities(_supported(**changes))

    def test_reports_all_conflicts_in_one_startup_error(self):
        with self.assertRaises(FeatureCaptureCapabilityError) as raised:
            validate_capabilities(
                _supported(device="cpu", pp_size=2, model_sites=())
            )
        message = str(raised.exception)
        self.assertIn("device='cpu'", message)
        self.assertIn("pp_size=2", message)
        self.assertIn("no verified 'block_output' adapter", message)

    def test_process_switch_defaults_off_and_can_be_overridden(self):
        with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(False):
            self.assertFalse(is_enabled())
            with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(True):
                self.assertTrue(is_enabled())
            self.assertFalse(is_enabled())


if __name__ == "__main__":
    unittest.main()
