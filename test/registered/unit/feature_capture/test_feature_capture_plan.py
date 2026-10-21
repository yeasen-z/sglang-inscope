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
"""Row-mapping tests for the feature-capture plan.

No GPU and no torch: the mapping is arithmetic over the packed batch, and the
cases that matter (a non-capturing request between two capturing ones) are
exactly the ones that are easy to get wrong and impossible to spot later.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.feature_capture.plan import (
    build_entries,
    build_slot_plan,
    group_by_slot,
    layout_slots,
)
from sglang.test.test_utils import CustomTestCase


def _spec(layers, positions):
    """Stand-in for FeatureRequestSpec: the plan only reads these two fields."""
    return SimpleNamespace(layers=tuple(layers), positions=tuple(positions))


class TestRowMapping(CustomTestCase):
    def test_offset_advances_over_non_capturing_requests(self):
        # A/B/C occupy 5/4/3 extend rows; only A and C capture their last token.
        # Flat token axis: A=0..4, B=5..8, C=9..11. A's last token is row 4,
        # C's last token is row 11. A wrong implementation that skips B's rows
        # would read row 4 and row 7.
        entries = build_entries(
            rids=["A", "B", "C"],
            specs=[_spec([4], [4]), None, _spec([4], [10])],
            extend_lens=[5, 4, 3],
            seq_lens=[5, 4, 11],  # C continues from an 8-token prefix
        )
        self.assertEqual([entry.rid for entry in entries], ["A", "C"])
        self.assertEqual(entries[0].row_start, 0)
        self.assertEqual(entries[0].positions, (4,))
        self.assertEqual(entries[1].row_start, 9)  # 5 + 4
        self.assertEqual(entries[1].positions, (10,))

        slots = layout_slots(entries)
        self.assertEqual(len(slots), 1)
        # rows are read from the forward's flat token axis
        self.assertEqual(slots[0].rows, (4, 11))

    def test_positions_outside_this_chunk_are_dropped(self):
        # Prompt of 20 tokens delivered as one 12-token chunk then an 8-token
        # chunk; position 15 only exists in the second forward.
        first = build_entries(
            rids=["A"],
            specs=[_spec([4], [15])],
            extend_lens=[12],
            seq_lens=[12],
        )
        self.assertEqual(first, ())

        second = build_entries(
            rids=["A"],
            specs=[_spec([4], [15])],
            extend_lens=[8],
            seq_lens=[20],
        )
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].position_start, 12)
        self.assertEqual(second[0].positions, (15,))

    def test_empty_batch_captures_nothing(self):
        self.assertEqual(
            build_entries(rids=[], specs=[], extend_lens=[], seq_lens=[]), ()
        )

    def test_no_specs_captures_nothing(self):
        self.assertEqual(
            build_entries(
                rids=["A", "B"], specs=[None, None], extend_lens=[3, 4], seq_lens=[3, 4]
            ),
            (),
        )

    def test_decode_step_maps_single_row(self):
        entries = build_entries(
            rids=["A"], specs=[_spec([4], [41])], extend_lens=[1], seq_lens=[42]
        )
        self.assertEqual(entries[0].row_start, 0)
        self.assertEqual(entries[0].positions, (41,))

    def test_inconsistent_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            build_entries(rids=["A"], specs=[_spec([4], [0])], extend_lens=[1], seq_lens=[1, 2])
        with self.assertRaises(ValueError):
            build_entries(rids=["A"], specs=[], extend_lens=[1], seq_lens=[1])
        with self.assertRaises(ValueError):
            build_entries(
                rids=["A"],
                specs=[_spec([4], [0])],
                extend_lens=[1],
                seq_lens=[1],
                prefix_lens=[0, 0],
            )

    def test_prefix_lens_takes_precedence(self):
        # ForwardBatch.extend_prefix_lens_cpu is the quantity we actually want;
        # when supplied it must win over the seq_lens - extend_lens derivation.
        entries = build_entries(
            rids=["A"],
            specs=[_spec([4], [10])],
            extend_lens=[4],
            seq_lens=[14],
            prefix_lens=[10],
        )
        self.assertEqual(entries[0].position_start, 10)
        self.assertEqual(entries[0].positions, (10,))


class TestSlotLayout(CustomTestCase):
    def test_complete_plan_skips_decode_and_batches_without_specs(self):
        self.assertEqual(
            build_slot_plan(
                rids=["A"],
                specs=[_spec([4], [3])],
                extend_lens=None,
                seq_lens=[4],
            ),
            (),
        )
        self.assertEqual(
            build_slot_plan(
                rids=["A"], specs=[None], extend_lens=[4], seq_lens=[4]
            ),
            (),
        )

    def test_layers_are_grouped_and_disjoint(self):
        entries = build_entries(
            rids=["A", "B"],
            specs=[_spec([4, 8], [3]), _spec([4], [7])],
            extend_lens=[5, 8],
            seq_lens=[5, 8],
        )
        slots = layout_slots(entries)
        self.assertEqual([slot.layer for slot in slots], [4, 8])

        layer4, layer8 = slots
        # Layer 4 serves both requests, in batch order.
        self.assertEqual(layer4.segments, (("A", (3,)), ("B", (7,))))
        self.assertEqual(layer4.rows, (3, 5 + 7))
        self.assertEqual(layer8.segments, (("A", (3,)),))
        self.assertEqual(layer8.rows, (3,))

        # Buffers must not overlap: a shared offset would corrupt one slot.
        self.assertEqual(layer4.out_start, 0)
        self.assertEqual(layer8.out_start, layer4.out_stop)
        self.assertEqual(layer8.out_stop, 3)

    def test_multi_position_rows_are_ordered(self):
        entries = build_entries(
            rids=["A"],
            specs=[_spec([4], [2, 5, 7])],
            extend_lens=[8],
            seq_lens=[8],
        )
        slots = layout_slots(entries)
        self.assertEqual(slots[0].rows, (2, 5, 7))
        self.assertEqual(slots[0].segments, (("A", (2, 5, 7)),))

    def test_multi_layer_multi_request_total_rows(self):
        entries = build_entries(
            rids=["A", "B", "C"],
            specs=[_spec([4, 8], [1, 2]), None, _spec([8], [9])],
            extend_lens=[3, 2, 10],
            seq_lens=[3, 2, 10],
        )
        by_layer, total = group_by_slot(entries)
        self.assertEqual(sorted(by_layer), [4, 8])
        # A contributes 2 layers x 2 positions = 4, C 1 layer x 1 position = 1.
        self.assertEqual(total, 5)
        slots = layout_slots(entries)
        self.assertEqual(sum(len(slot.rows) for slot in slots), total)


if __name__ == "__main__":
    unittest.main()
