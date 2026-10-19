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
"""Request-contract tests for the feature-capture channel.

These run without a GPU and without torch: the contract is validated in the
tokenizer process, so it must not need any of the inference stack.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.feature_capture.spec import (
    FeatureCaptureError,
    FeatureCaptureSuffixNotFound,
    validate,
)
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 36
HIDDEN = 4096
PROMPT = tuple(range(100, 130))  # 30 tokens, distinct ids


def _validate(raw, **kwargs):
    params = dict(num_layers=NUM_LAYERS, hidden_size=HIDDEN, prompt_ids=PROMPT)
    params.update(kwargs)
    return validate(raw, **params)


class TestLayerResolution(CustomTestCase):
    """Layer ids index the HF hidden_states tuple, with no off-by-one anywhere."""

    def test_last_layer_is_num_layers(self):
        for requested in (NUM_LAYERS, -1):
            with self.subTest(requested=requested):
                spec = _validate({"layers": [requested]})
                self.assertEqual(spec.layers, (NUM_LAYERS,))

    def test_first_layer_is_embedding_output(self):
        spec = _validate({"layers": [0]})
        self.assertEqual(spec.layers, (0,))

    def test_negative_layers_count_back_from_final(self):
        spec = _validate({"layers": [-2]})
        self.assertEqual(spec.layers, (NUM_LAYERS - 1,))

    def test_range_is_inclusive_of_num_layers(self):
        spec = _validate({"layers": [0, NUM_LAYERS]})
        self.assertEqual(spec.layers, (0, NUM_LAYERS))

    def test_rejects_out_of_range(self):
        for bad in (NUM_LAYERS + 1, -(NUM_LAYERS + 2)):
            with self.subTest(bad=bad):
                with self.assertRaises(FeatureCaptureError):
                    _validate({"layers": [bad]})

    def test_rejects_duplicate_after_resolution(self):
        # 36 and -1 name the same slot; silently capturing it twice would
        # double-write the same output row.
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": [NUM_LAYERS, -1]})

    def test_rejects_bool_and_float(self):
        for bad in (True, 3.0):
            with self.subTest(bad=bad):
                with self.assertRaises(FeatureCaptureError):
                    _validate({"layers": [bad]})

    def test_rejects_empty_and_oversized(self):
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": []})
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": list(range(9))})

    def test_preserves_caller_order(self):
        spec = _validate({"layers": [20, 4, 16]})
        self.assertEqual(spec.layers, (20, 4, 16))


class TestPositionModes(CustomTestCase):
    def test_last_is_final_prompt_token(self):
        spec = _validate({"layers": [4], "positions": {"mode": "last"}})
        self.assertEqual(spec.positions, (len(PROMPT) - 1,))
        self.assertEqual(spec.position_mode, "last")

    def test_default_mode_is_last(self):
        spec = _validate({"layers": [4]})
        self.assertEqual(spec.position_mode, "last")
        self.assertEqual(spec.positions, (len(PROMPT) - 1,))

    def test_all_covers_every_token(self):
        spec = _validate({"layers": [4], "positions": {"mode": "all"}})
        self.assertEqual(spec.positions, tuple(range(len(PROMPT))))

    def test_explicit_positive(self):
        spec = _validate(
            {"layers": [4], "positions": {"mode": "explicit", "index": [17]}}
        )
        self.assertEqual(spec.positions, (17,))

    def test_explicit_negative_resolves_from_end(self):
        spec = _validate(
            {"layers": [4], "positions": {"mode": "explicit", "index": [-1]}}
        )
        self.assertEqual(spec.positions, (len(PROMPT) - 1,))

    def test_explicit_keeps_order_and_collapses_duplicates(self):
        # -1 and len-1 are the same token: one output row, not two.
        spec = _validate(
            {
                "layers": [4],
                "positions": {"mode": "explicit", "index": [12, 3, len(PROMPT) - 1, -1]},
            }
        )
        self.assertEqual(spec.positions, (12, 3, len(PROMPT) - 1))

    def test_explicit_rejects_out_of_range(self):
        for bad in ([len(PROMPT)], [-len(PROMPT) - 1]):
            with self.subTest(bad=bad):
                with self.assertRaises(FeatureCaptureError):
                    _validate({"layers": [4], "positions": {"mode": "explicit", "index": bad}})

    def test_explicit_rejects_empty(self):
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": [4], "positions": {"mode": "explicit", "index": []}})


class TestSuffixMode(CustomTestCase):
    """suffix_last is a validator, not an alias for last."""

    def test_takes_last_token_of_suffix(self):
        suffix = list(PROMPT[-3:])
        spec = _validate(
            {
                "layers": [4],
                "positions": {"mode": "suffix_last", "suffix_token_ids": suffix},
            }
        )
        self.assertEqual(spec.positions, (len(PROMPT) - 1,))
        self.assertEqual(spec.suffix_occurrences, 1)

    def test_missing_suffix_is_fatal_not_a_fallback(self):
        # A prompt that no longer ends where the caller believes must fail
        # loudly; falling back to `last` would hand the probe a wrong vector.
        with self.assertRaises(FeatureCaptureSuffixNotFound):
            _validate(
                {
                    "layers": [4],
                    "positions": {"mode": "suffix_last", "suffix_token_ids": [999, 998]},
                }
            )

    def test_suffix_not_at_the_end_still_resolves_to_its_own_last_token(self):
        suffix = list(PROMPT[5:8])
        spec = _validate(
            {
                "layers": [4],
                "positions": {"mode": "suffix_last", "suffix_token_ids": suffix},
            }
        )
        self.assertEqual(spec.positions, (7,))
        self.assertNotEqual(spec.positions, (len(PROMPT) - 1,))

    def test_uses_last_occurrence_and_counts_it(self):
        prompt = (1, 2, 3, 4, 1, 2, 3)
        spec = validate(
            {"layers": [4], "positions": {"mode": "suffix_last", "suffix_token_ids": [1, 2, 3]}},
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            prompt_ids=prompt,
        )
        self.assertEqual(spec.positions, (6,))
        self.assertEqual(spec.suffix_occurrences, 2)

    def test_suffix_text_requires_pretokenized_ids(self):
        with self.assertRaises(FeatureCaptureError):
            _validate(
                {"layers": [4], "positions": {"mode": "suffix_last", "suffix_text": "abc"}}
            )

    def test_suffix_text_uses_caller_tokenized_ids(self):
        spec = _validate(
            {"layers": [4], "positions": {"mode": "suffix_last", "suffix_text": "abc"}},
            suffix_token_ids=list(PROMPT[-2:]),
        )
        self.assertEqual(spec.positions, (len(PROMPT) - 1,))

    def test_rejects_empty_suffix(self):
        with self.assertRaises(FeatureCaptureError):
            _validate(
                {"layers": [4], "positions": {"mode": "suffix_last", "suffix_token_ids": []}}
            )


class TestContractRejection(CustomTestCase):
    def test_unknown_top_level_key(self):
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": [4], "layer": [4]})

    def test_unknown_position_key(self):
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": [4], "positions": {"mode": "last", "index": [1]}})

    def test_unsupported_site(self):
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": [4], "site": "ffn_output"})

    def test_unsupported_encoding(self):
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": [4], "encoding": "float16"})

    def test_unknown_mode(self):
        with self.assertRaises(FeatureCaptureError):
            _validate({"layers": [4], "positions": {"mode": "middle"}})

    def test_byte_budget(self):
        # 8 layers x 30 positions x 4096 x 4 bytes is well over a 1 MiB budget.
        with self.assertRaises(FeatureCaptureError):
            _validate(
                {"layers": list(range(8)), "positions": {"mode": "all"}},
                max_bytes=1024 * 1024,
            )


if __name__ == "__main__":
    unittest.main()
