"""Pure feature collection must bypass the vocabulary projection only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import torch

from sglang.srt.layers import logits_processor as module
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)


def batch(**updates):
    value = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor([1]),
        req_pool_indices=None,
        seq_lens=None,
        out_cache_loc=None,
        seq_lens_sum=1,
    )
    defaults = dict(
        is_prefill_only=True,
        feature_capture_plan=(object(),),
        feature_capture_requested=True,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_logprob=False,
        return_pooled_hidden_states=False,
        return_hidden_states_before_norm=False,
        extend_seq_lens_cpu=[1],
        extend_logprob_start_lens_cpu=[0],
        top_logprobs_nums=[0],
        token_ids_logprobs=[None],
    )
    defaults.update(updates)
    for key, item in defaults.items():
        setattr(value, key, item)
    return value


class TestFeatureOnlyLogits(unittest.TestCase):
    def setUp(self):
        self.processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(self.processor)

    def test_pure_capture_does_not_enter_lm_head_path(self):
        with patch.object(self.processor, "_get_pruned_states") as prune:
            result = self.processor.forward(None, None, None, batch())
        self.assertIsNone(result.next_token_logits)
        prune.assert_not_called()

    def test_chunk_without_capture_rows_still_avoids_logits(self):
        result = self.processor.forward(
            None, None, None, batch(feature_capture_plan=())
        )
        self.assertIsNone(result.next_token_logits)

    def test_generation_and_output_consumers_keep_ordinary_path(self):
        changes = (
            dict(is_prefill_only=False),
            dict(feature_capture_requested=False),
            dict(return_logprob=True),
            dict(capture_hidden_mode=CaptureHiddenMode.FULL),
            dict(return_pooled_hidden_states=True),
            dict(return_hidden_states_before_norm=True),
            dict(sampling_info=SimpleNamespace(grammars=[object()])),
        )
        for update in changes:
            with self.subTest(update=update):
                value = batch(**update)
                self.assertFalse(
                    LogitsMetadata.from_forward_batch(value).feature_capture_only
                )
                with (
                    patch.object(
                        module.layernorm_sp,
                        "maybe_exit_gather",
                        return_value=(None, None),
                    ),
                    patch.object(
                        self.processor,
                        "_get_pruned_states",
                        side_effect=RuntimeError("ordinary path"),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "ordinary path"):
                        self.processor.forward(None, None, None, value)

    def test_setwise_scoring_still_projects_vocabulary(self):
        value = batch(token_indices_to_pool=[0])
        with (
            patch.object(
                module.layernorm_sp, "maybe_exit_gather", return_value=(None, None)
            ),
            patch.object(
                self.processor, "compute_logprobs_at_positions", return_value="scoring"
            ) as score,
        ):
            self.assertEqual(self.processor.forward(None, None, None, value), "scoring")
        score.assert_called_once()

    def test_multi_item_scoring_still_projects_vocabulary(self):
        value = batch(multi_item_delimiter_indices=[0])
        with (
            patch.object(
                module.layernorm_sp, "maybe_exit_gather", return_value=(None, None)
            ),
            patch.object(
                self.processor,
                "compute_logprobs_for_multi_item_scoring",
                return_value="scoring",
            ) as score,
        ):
            self.assertEqual(self.processor.forward(None, None, None, value), "scoring")
        score.assert_called_once()

    def test_generation_metadata_has_no_capture_optimization_by_default(self):
        self.assertFalse(
            LogitsMetadata(forward_mode=ForwardMode.EXTEND).feature_capture_only
        )


if __name__ == "__main__":
    unittest.main()
