"""Opt-in boundary cutoff for dedicated, fixed-layer feature servers."""
import os


def configured_layer():
    value = int(os.environ.get("SGLANG_FEATURE_EARLY_EXIT_LAYER", "0"))
    return value if value > 0 else None


def resolve_layer(batch, capture_hidden_mode, return_hidden_states_before_norm):
    cutoff = configured_layer()
    if cutoff is None or not batch.is_prefill_only or not batch.reqs:
        return None
    if (batch.return_logprob or int(capture_hidden_mode) != 0
            or return_hidden_states_before_norm
            or any(req.return_pooled_hidden_states or req.grammar is not None
                   for req in batch.reqs)):
        return None
    specs = [req.feature_capture for req in batch.reqs]
    if any(spec is None or max(spec.layers) != cutoff for spec in specs):
        return None
    return cutoff
