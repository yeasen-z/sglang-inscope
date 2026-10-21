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
"""Request normalization tests for the feature-capture IPC field."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from array import array
from types import SimpleNamespace

import msgspec

from sglang.srt.environ import envs
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.msgpack_utils import dec_hook, enc_hook, ext_hook
from sglang.test.test_utils import CustomTestCase


class TestFeatureCaptureRequestNormalization(CustomTestCase):
    def test_shared_batch_spec_is_copied_per_request(self):
        request = GenerateReqInput(
            text=["first", "second"],
            feature_capture={"layers": [-1], "positions": {"mode": "last"}},
        )
        request.normalize_batch_and_arguments()

        first = request[0].feature_capture
        second = request[1].feature_capture
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        first["layers"].append(3)
        self.assertEqual(second["layers"], [-1])

    def test_per_request_specs_preserve_opt_out(self):
        request = GenerateReqInput(
            text=["capture", "ordinary"],
            feature_capture=[{"layers": [4]}, None],
        )
        request.normalize_batch_and_arguments()
        self.assertEqual(request[0].feature_capture, {"layers": [4]})
        self.assertIsNone(request[1].feature_capture)

    def test_batch_length_mismatch_is_rejected(self):
        request = GenerateReqInput(
            text=["first", "second"], feature_capture=[{"layers": [4]}]
        )
        with self.assertRaisesRegex(ValueError, "equal the batch size"):
            request.normalize_batch_and_arguments()

    def test_single_request_rejects_batch_shaped_capture(self):
        request = GenerateReqInput(
            text="one", feature_capture=[{"layers": [4]}]
        )
        with self.assertRaisesRegex(ValueError, "mapping for one request"):
            request.normalize_batch_and_arguments()


class TestFeatureCaptureEngineAPI(CustomTestCase):
    def test_engine_apis_forward_capture_contract(self):
        """The public Engine API must not silently drop the capture request."""
        seen = []

        class FakeTokenizerManager:
            async def generate_request(self, request, _raw_request):
                seen.append(request)
                yield {"meta_info": {"features": []}}

        engine = object.__new__(Engine)
        engine.tokenizer_manager = FakeTokenizerManager()
        engine._resolve_routed_dp_rank = lambda routed, deprecated: routed
        engine.loop = SimpleNamespace(run_until_complete=asyncio.run)

        capture = {"layers": [0, -1]}
        sync_result = engine.generate(
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 0},
            feature_capture=capture,
        )

        async_result = asyncio.run(
            engine.async_generate(
                input_ids=[1, 2, 3],
                sampling_params={"max_new_tokens": 0},
                feature_capture=capture,
            )
        )

        expected = {"meta_info": {"features": []}}
        self.assertEqual(sync_result, expected)
        self.assertEqual(async_result, expected)
        self.assertEqual(
            [request.feature_capture for request in seen], [capture, capture]
        )


class TestFeatureCaptureTokenizerBoundary(CustomTestCase):
    def setUp(self):
        self.manager = object.__new__(TokenizerManager)
        self.manager.model_config = SimpleNamespace(
            num_hidden_layers=36, hidden_size=4096
        )
        self.manager.tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [8, 9]
        )

    def test_disabled_process_rejects_capture_request(self):
        request = GenerateReqInput(
            text="prompt", feature_capture={"layers": [-1]}
        )
        with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(False):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_ENABLE_FEATURE_CAPTURE=1"
            ):
                self.manager._validate_feature_capture(
                    request, input_ids=[1, 2, 3], mm_inputs=None
                )

    def test_enabled_process_resolves_layer_and_position_before_ipc(self):
        request = GenerateReqInput(
            text="prompt", feature_capture={"layers": [-1]}
        )
        with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(True):
            spec = self.manager._validate_feature_capture(
                request, input_ids=[1, 2, 3], mm_inputs=None
            )
        self.assertEqual(spec.layers, (36,))
        self.assertEqual(spec.positions, (2,))

    def test_session_request_is_rejected_before_positions_can_shift(self):
        request = GenerateReqInput(
            text="prompt",
            session_id="conversation",
            feature_capture={"layers": [-1]},
        )
        with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(True):
            with self.assertRaisesRegex(ValueError, "does not support session"):
                self.manager._validate_feature_capture(
                    request, input_ids=[1, 2, 3], mm_inputs=None
                )

    def test_resolved_spec_survives_scheduler_request_construction(self):
        request = GenerateReqInput(
            text="prompt", feature_capture={"layers": [-1]}
        )
        with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(True):
            spec = self.manager._validate_feature_capture(
                request, input_ids=[1, 2, 3], mm_inputs=None
            )
        req = Req(
            rid="request-id",
            origin_input_text="prompt",
            origin_input_ids=array("q", [1, 2, 3]),
            sampling_params=SamplingParams(max_new_tokens=0),
            feature_capture=spec,
        )
        self.assertIs(req.feature_capture, spec)

    def test_capture_limits_prefix_cache_to_earliest_incomplete_position(self):
        request = GenerateReqInput(
            text="prompt",
            feature_capture={
                "layers": [1, 2],
                "positions": {"mode": "explicit", "index": [5, 10]},
            },
        )
        with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(True):
            spec = self.manager._validate_feature_capture(
                request, input_ids=list(range(21)), mm_inputs=None
            )
        req = Req(
            rid="request-id",
            origin_input_text="prompt",
            origin_input_ids=array("q", range(21)),
            sampling_params=SamplingParams(max_new_tokens=0),
            feature_capture=spec,
        )

        self.assertEqual(req._compute_max_prefix_len(21), 5)
        req.feature_capture_values[(1, 5)] = object()
        self.assertEqual(req._compute_max_prefix_len(21), 5)
        req.feature_capture_values[(2, 5)] = object()
        self.assertEqual(req._compute_max_prefix_len(21), 10)
        req.feature_capture_values[(1, 10)] = object()
        req.feature_capture_values[(2, 10)] = object()
        self.assertEqual(req._compute_max_prefix_len(21), 20)

    def test_resolved_spec_round_trips_through_tokenizer_scheduler_ipc(self):
        request = GenerateReqInput(
            text="prompt", feature_capture={"layers": [-1]}
        )
        with envs.SGLANG_ENABLE_FEATURE_CAPTURE.override(True):
            spec = self.manager._validate_feature_capture(
                request, input_ids=[1, 2, 3], mm_inputs=None
            )
        tokenized = TokenizedGenerateReqInput(
            input_text="prompt",
            input_ids=array("q", [1, 2, 3]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(max_new_tokens=0),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            rid="request-id",
            feature_capture=spec,
        )
        payload = msgspec.msgpack.encode(tokenized, enc_hook=enc_hook)
        restored = msgspec.msgpack.decode(
            payload,
            type=TokenizedGenerateReqInput,
            dec_hook=dec_hook,
            ext_hook=ext_hook,
        )
        self.assertEqual(restored.feature_capture, spec)


if __name__ == "__main__":
    unittest.main()
