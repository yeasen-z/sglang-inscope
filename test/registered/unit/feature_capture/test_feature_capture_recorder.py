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
"""CPU tests for device recording and host-side request attribution."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
import base64
from types import SimpleNamespace

import torch

from sglang.srt.feature_capture.plan import SlotPlan
from sglang.srt.feature_capture.encoding import build_feature_records
from sglang.srt.feature_capture.recorder import FeatureCaptureRecorder
from sglang.srt.feature_capture.spec import validate
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.test.test_utils import CustomTestCase


class TestFeatureCaptureRecorder(CustomTestCase):
    def test_features_are_request_level_metadata(self):
        self.assertTrue(
            SchedulerOutputStreamer.is_request_level_customized_info_key(
                None, "features"
            )
        )
        self.assertFalse(
            SchedulerOutputStreamer.is_request_level_customized_info_key(
                None, "ordinary-token-field"
            )
        )

    def test_float_list_encoding_converts_model_dtype_to_json_numbers(self):
        spec = validate(
            {"layers": [4], "encoding": "float_list"},
            num_layers=4,
            hidden_size=2,
            prompt_ids=[1],
        )
        records = build_feature_records(
            spec=spec,
            values={(4, 0): torch.tensor([1.5, -2.0], dtype=torch.bfloat16)},
        )
        self.assertEqual(records[0]["dtype"], "bfloat16")
        self.assertEqual(records[0]["data"], [[1.5, -2.0]])

    def test_selected_rows_are_attributed_by_layer_position_and_request(self):
        plan = (
            SlotPlan(
                layer=4,
                rows=(1, 3),
                segments=(("A", (7,)), ("B", (11,))),
                out_start=0,
            ),
            SlotPlan(
                layer=8,
                rows=(0,),
                segments=(("A", (6,)),),
                out_start=2,
            ),
        )
        recorder = FeatureCaptureRecorder(plan)
        hidden4 = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        hidden8 = hidden4 + 100
        recorder.capture_hook(layer=4)(hidden4, owned=True)
        recorder.capture_hook(layer=8)(hidden8)

        host = recorder.device_output().copy_to_host(lambda tensor: tensor.clone())
        req_a = SimpleNamespace(
            rid="A",
            is_retracted=False,
            feature_capture=validate(
                {
                    "layers": [4, 8],
                    "positions": {"mode": "explicit", "index": [7, 6]},
                },
                num_layers=8,
                hidden_size=4,
                prompt_ids=list(range(12)),
            ),
            feature_capture_values={},
            customized_info=None,
        )
        req_b = SimpleNamespace(
            rid="B",
            is_retracted=False,
            feature_capture=validate(
                {
                    "layers": [4],
                    "positions": {"mode": "explicit", "index": [11]},
                },
                num_layers=8,
                hidden_size=4,
                prompt_ids=list(range(12)),
            ),
            feature_capture_values={},
            customized_info=None,
        )
        host.consume(SimpleNamespace(reqs=[req_a, req_b]), commits=[None, None])

        torch.testing.assert_close(req_a.feature_capture_values[(4, 7)], hidden4[1])
        torch.testing.assert_close(req_b.feature_capture_values[(4, 11)], hidden4[3])
        torch.testing.assert_close(req_a.feature_capture_values[(8, 6)], hidden8[0])
        self.assertEqual(
            [record["layer"] for record in req_a.customized_info["features"]],
            [4, 8],
        )
        record = req_a.customized_info["features"][0]
        self.assertEqual(record["shape"], [1, 4])
        self.assertEqual(
            base64.b64decode(record["data"]), hidden4[1].numpy().tobytes()
        )

    def test_retracted_request_is_skipped_and_existing_cell_is_not_overwritten(self):
        plan = (
            SlotPlan(
                layer=4,
                rows=(0, 1),
                segments=(("A", (2,)), ("B", (3,))),
                out_start=0,
            ),
        )
        recorder = FeatureCaptureRecorder(plan)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        recorder.capture_hook(layer=4)(hidden)
        host = recorder.device_output().copy_to_host(lambda tensor: tensor.clone())
        original = torch.tensor([9.0, 9.0])
        req_a = SimpleNamespace(
            rid="A",
            is_retracted=False,
            feature_capture_values={(4, 2): original},
            feature_capture=validate(
                {"layers": [4], "positions": {"mode": "explicit", "index": [2]}},
                num_layers=4,
                hidden_size=2,
                prompt_ids=list(range(4)),
            ),
            customized_info=None,
        )
        req_b = SimpleNamespace(
            rid="B",
            is_retracted=True,
            feature_capture=None,
            feature_capture_values={},
            customized_info=None,
        )
        host.consume(SimpleNamespace(reqs=[req_a, req_b]), commits=[None, None])
        self.assertIs(req_a.feature_capture_values[(4, 2)], original)
        self.assertEqual(req_b.feature_capture_values, {})

    def test_missing_model_hook_fails_instead_of_returning_partial_features(self):
        recorder = FeatureCaptureRecorder(
            (
                SlotPlan(
                    layer=4,
                    rows=(0,),
                    segments=(("A", (0,)),),
                    out_start=0,
                ),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "did not run"):
            recorder.device_output()



class TestFeatureEncodingParity(CustomTestCase):
    def test_base64_preserves_model_dtype_and_noncontiguous_values(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                values = torch.arange(24, dtype=dtype).reshape(4, 6)[:, ::2]
                self.assertFalse(values.is_contiguous())
                spec = validate(
                    {"layers": [4], "encoding": "base64"},
                    num_layers=4, hidden_size=3, prompt_ids=list(range(4)),
                )
                records = build_feature_records(
                    spec=spec, values={(4, position): values[position] for position in range(4)},
                )
                record = records[0]
                expected = torch.stack([values[position] for position in record["positions"]])
                self.assertEqual(record["dtype"], str(dtype).removeprefix("torch."))
                self.assertEqual(
                    record["data"],
                    base64.b64encode(expected.view(torch.uint8).numpy().tobytes()).decode("ascii"),
                )

if __name__ == "__main__":
    unittest.main()
