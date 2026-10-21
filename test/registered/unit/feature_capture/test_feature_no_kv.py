"""Dedicated feature-only no-KV mode preserves native safety constraints."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups import kv_cache_hook
from sglang.srt.layers.attention import flashattention_backend as fa
from sglang.srt.managers import tokenizer_manager as tokenizer
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode


def config(**updates):
    values = dict(
        prefill_only_disable_kv_cache=True,
        is_embedding=False,
        kv_cache_dtype="auto",
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        attn_cp_size=1,
        enable_prefill_cp=False,
        enable_hisparse=False,
        attention_backend="fa3",
        enable_priority_scheduling=False,
        default_priority_value=0,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def validate(cfg, enabled):
    switch = SimpleNamespace(
        SGLANG_ENABLE_FEATURE_CAPTURE=SimpleNamespace(get=lambda: enabled)
    )
    with (
        patch.object(kv_cache_hook, "resolving_view", return_value=cfg),
        patch.object(kv_cache_hook, "resolved_view", return_value=cfg),
        patch.object(kv_cache_hook, "envs", switch),
    ):
        kv_cache_hook.validate_prefill_only_disable_kv_cache_args(cfg)


class TestFeatureNoKvConfiguration(unittest.TestCase):
    def test_embedding_mode_retains_existing_behavior(self):
        validate(config(is_embedding=True), False)

    def test_generation_model_requires_native_capture_opt_in(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_ENABLE_FEATURE_CAPTURE"):
            validate(config(), False)
        validate(config(), True)

    def test_unsupported_configurations_still_fail(self):
        changes = (
            dict(kv_cache_dtype="nvfp4"),
            dict(kv_cache_dtype="fp4_mx_block16"),
            dict(kv_cache_dtype="mxfp8"),
            dict(chunked_prefill_size=128),
            dict(disable_radix_cache=False),
            dict(attn_cp_size=2),
            dict(enable_prefill_cp=True),
            dict(enable_hisparse=True),
        )
        for update in changes:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate(config(**update), True)

    def test_attention_backend_constraint_remains(self):
        cfg = config()
        with (
            patch.object(kv_cache_hook, "resolving_view", return_value=cfg),
            patch.object(kv_cache_hook, "resolved_view", return_value=cfg),
            patch.object(
                kv_cache_hook,
                "attention_backends_of",
                return_value=("triton", "triton"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "FA prefill backend"):
                kv_cache_hook.handle_prefill_only_disable_kv_cache(cfg)

    def test_raw_kv_path_requires_dedicated_or_embedding_mode(self):
        cases = (
            (True, False, False, -1, True, False, True),
            (False, True, True, -1, True, False, True),
            (False, False, True, -1, True, False, False),
            (False, True, False, -1, True, False, False),
            (False, True, True, 128, True, False, False),
            (False, True, True, -1, False, False, False),
            (False, True, True, -1, True, True, False),
        )
        for embedding, no_kv, capture, chunk, no_radix, mla, expected in cases:
            cfg = config(
                is_embedding=embedding,
                prefill_only_disable_kv_cache=no_kv,
                chunked_prefill_size=chunk,
                disable_radix_cache=no_radix,
            )
            with (
                self.subTest(case=(embedding, no_kv, capture, chunk, no_radix, mla)),
                patch.object(fa, "get_model", return_value=cfg),
                patch.object(fa, "get_schedule", return_value=cfg),
                patch.object(fa, "get_memory", return_value=cfg),
                patch.object(fa, "feature_capture_enabled", return_value=capture),
            ):
                self.assertEqual(fa._use_raw_prefill_kv(use_mla=mla), expected)


class TestFeatureNoKvRequests(unittest.TestCase):
    def setUp(self):
        self.manager = tokenizer.TokenizerManager.__new__(tokenizer.TokenizerManager)
        self.manager.is_generation = True
        self.manager.context_len = 100
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = False
        self.manager._validate_joint_schema_request = Mock()
        self.manager._validate_token_ids_logprob = Mock()
        self.manager._validate_for_matryoshka_dim = Mock()
        self.cfg = config()
        self.schedule = patch.object(tokenizer, "get_schedule", return_value=self.cfg)
        self.execution = patch.object(
            tokenizer,
            "get_exec",
            return_value=SimpleNamespace(
                features=SimpleNamespace(
                    enable_encoder_swa_bounded_replay=False,
                    enable_custom_logit_processor=False,
                )
            ),
        )
        self.hidden = patch.object(
            tokenizer,
            "get_server_return_hidden_states_mode",
            return_value=CaptureHiddenMode.NULL,
        )
        for handle in (self.schedule, self.execution, self.hidden):
            handle.start()
            self.addCleanup(handle.stop)

    def request(self, **updates):
        values = dict(
            input_ids=[1],
            sampling_params={"max_new_tokens": 0},
            feature_capture={"layers": [8], "positions": {"mode": "last"}},
        )
        values.update(updates)
        return GenerateReqInput(**values)

    def test_model_initialization_defers_output_budget_to_scheduler(self):
        self.manager.server_args = self.cfg
        self.manager.model_config_class = SimpleNamespace(
            from_server_args=lambda args: SimpleNamespace(
                is_generation=True, context_len=100, image_token_id=None
            )
        )
        with (
            patch.object(
                tokenizer, "get_model", return_value=SimpleNamespace(model_path="test")
            ),
            patch.object(
                tokenizer,
                "get_serving",
                return_value=SimpleNamespace(served_model_name="test"),
            ),
            patch.object(tokenizer, "compute_num_reserved_tokens", return_value=0),
        ):
            self.manager.init_model_config()
        self.assertFalse(self.manager.validate_total_tokens)

    def test_explicit_feature_only_request_is_allowed(self):
        self.manager._validate_one_request(self.request(), [1])

    def test_non_capture_and_generation_requests_fail_before_normalization(self):
        cases = (
            self.request(feature_capture=None),
            self.request(sampling_params={}),
            self.request(sampling_params={"max_new_tokens": 1}),
            self.request(sampling_params={"max_new_tokens": None}),
            EmbeddingReqInput(input_ids=[1]),
        )
        for request in cases:
            with (
                self.subTest(request=request),
                self.assertRaisesRegex(ValueError, "feature-only service"),
            ):
                self.manager._validate_one_request(request, [1])

    def test_normal_generation_service_remains_allowed(self):
        self.cfg.prefill_only_disable_kv_cache = False
        self.manager._validate_one_request(
            self.request(feature_capture=None, sampling_params={"max_new_tokens": 20}),
            [1],
        )

    def test_embedding_service_is_unchanged(self):
        self.manager.is_generation = False
        self.manager._validate_one_request(
            EmbeddingReqInput(input_ids=[1], sampling_params={}), [1]
        )

    def test_output_budget_does_not_reject_or_mutate_valid_prompt(self):
        self.cfg.prefill_only_disable_kv_cache = False
        prompt = [1] * 99
        self.manager._validate_one_request(
            self.request(
                feature_capture=None, sampling_params={"max_new_tokens": 1000}
            ),
            prompt,
        )
        self.assertEqual(prompt, [1] * 99)

    def test_overlong_prompt_remains_rejected_without_truncation(self):
        prompt = [1] * 100
        with self.assertRaisesRegex(ValueError, "longer than"):
            self.manager._validate_one_request(self.request(), prompt)
        self.assertEqual(prompt, [1] * 100)


if __name__ == "__main__":
    unittest.main()
