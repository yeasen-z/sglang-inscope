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
"""Callback-composition tests for the model-facing feature-capture bridge."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.feature_capture.bridge import (
    final_capture,
    layer_capture,
    layer_capture_accumulator,
)
from sglang.test.test_utils import CustomTestCase


class _Recorder:
    def __init__(self, wanted_layers):
        self.wanted_layers = set(wanted_layers)
        self.calls = []

    def capture_hook(self, *, layer):
        if layer not in self.wanted_layers:
            return None

        def capture(value, *, owned=False):
            self.calls.append((layer, value, owned))

        return capture


class TestFeatureCaptureBridge(CustomTestCase):
    def test_accumulator_adapter_preserves_feature_and_auxiliary_consumers(self):
        recorder = _Recorder({3})

        class Auxiliary:
            def __init__(self):
                self.calls = []

            def capture(self, value, *, owned=False):
                self.calls.append((value, owned))

        auxiliary = Auxiliary()
        accumulator = layer_capture_accumulator(
            layer=3,
            recorder=recorder,
            auxiliary_accumulator=auxiliary,
        )
        accumulator.capture("hidden", owned=True)

        self.assertEqual(recorder.calls, [(3, "hidden", False)])
        self.assertEqual(auxiliary.calls, [("hidden", True)])

    def test_disabled_capture_preserves_existing_auxiliary_callback(self):
        calls = []

        def auxiliary(value, *, owned=False):
            calls.append((value, owned))

        hook = layer_capture(layer=3, recorder=None, auxiliary_hook=auxiliary)
        self.assertIs(hook, auxiliary)
        hook("hidden", owned=True)
        self.assertEqual(calls, [("hidden", True)])

    def test_shared_layer_calls_both_consumers_with_separate_ownership(self):
        recorder = _Recorder({3})
        auxiliary_calls = []

        def auxiliary(value, *, owned=False):
            auxiliary_calls.append((value, owned))

        hook = layer_capture(
            layer=3, recorder=recorder, auxiliary_hook=auxiliary
        )
        hook("hidden", owned=True)

        # The recorder must copy even when the producer gives ownership to the
        # auxiliary path; forwarding True here risks reading reused storage.
        self.assertEqual(recorder.calls, [(3, "hidden", False)])
        self.assertEqual(auxiliary_calls, [("hidden", True)])

    def test_unrequested_layer_keeps_auxiliary_callback_identity(self):
        recorder = _Recorder({8})

        def auxiliary(value, *, owned=False):
            pass

        hook = layer_capture(
            layer=3, recorder=recorder, auxiliary_hook=auxiliary
        )
        self.assertIs(hook, auxiliary)

    def test_final_capture_uses_hf_num_layers_slot(self):
        recorder = _Recorder({36})
        hook = final_capture(num_layers=36, recorder=recorder)
        hook("pre-norm", owned=True)
        self.assertEqual(recorder.calls, [(36, "pre-norm", False)])


if __name__ == "__main__":
    unittest.main()
