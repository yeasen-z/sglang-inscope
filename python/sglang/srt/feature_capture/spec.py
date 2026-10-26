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
"""Feature-capture request contract: validation and token-position resolution.

Model-agnostic and import-free: this module must stay loadable without torch so
the HTTP/tokenizer process can validate a request before any GPU work exists.

Layer numbering
---------------
A requested layer ``L`` names a residual-stream block boundary. For a model
with ``N`` decoder layers::

    L = 0                 embedding output, before decoder layer 0
    1 <= L <= N - 1      output of decoder layer L-1
    L = N                 output of the last decoder layer, before final norm

For ``L < N`` this equals Hugging Face ``outputs.hidden_states[L]``. Current
Transformers Qwen3 places the post-final-norm tensor in
``outputs.hidden_states[N]``, so the ``L = N`` pre-norm reference instead comes
from a forward hook on the last decoder block. ``-1`` aliases ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "inscope.feature_capture.v1"

SUPPORTED_SITES: Tuple[str, ...] = ("block_output",)
SUPPORTED_ENCODINGS: Tuple[str, ...] = ("base64", "float_list")
POSITION_MODES: Tuple[str, ...] = ("last", "explicit", "suffix_last", "all")

# Conservative FP32 byte estimate, matching the sibling capture runtimes.
DEFAULT_MAX_TARGETS = int(os.environ.get("SGLANG_FEATURE_CAPTURE_MAX_TARGETS", "8"))
if not 1 <= DEFAULT_MAX_TARGETS <= 64:
    raise ValueError("SGLANG_FEATURE_CAPTURE_MAX_TARGETS must lie in [1, 64]")
DEFAULT_MAX_BYTES = 128 * 1024**2


class FeatureCaptureError(ValueError):
    """A request asked for something the capture channel cannot honour."""


class FeatureCaptureSuffixNotFound(FeatureCaptureError):
    """``suffix_last`` could not locate its suffix in the tokenized prompt.

    This is deliberately fatal rather than a silent fallback to ``last``: the
    whole point of the suffix mode is to detect that the prompt no longer ends
    where the caller believes it does. Falling back would hand the probe a
    vector from the wrong position with no signal that anything went wrong.
    """


@dataclass(frozen=True)
class FeatureTarget:
    """One (layer, site) capture point, already normalized."""

    layer: int
    site: str = "block_output"


@dataclass(frozen=True)
class FeatureRequestSpec:
    """A validated request, with positions resolved to absolute indices."""

    targets: Tuple[FeatureTarget, ...]
    positions: Tuple[int, ...]
    position_mode: str
    encoding: str = "base64"
    # How many times the suffix occurred. >1 means the prompt contains the
    # marker more than once; the last occurrence was used. Recorded so that a
    # surprising prompt can be spotted later instead of silently accepted.
    suffix_occurrences: int = 0

    @property
    def layers(self) -> Tuple[int, ...]:
        """Resolved layers, ordered as the caller asked, without duplicates."""
        return tuple(target.layer for target in self.targets)


def _reject_unknown(mapping: Mapping[str, Any], allowed: Sequence[str], what: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise FeatureCaptureError(
            f"{what} does not accept {', '.join(unknown)}; "
            f"allowed keys are {', '.join(sorted(allowed))}"
        )


def _resolve_layer(layer: Any, num_layers: int) -> int:
    """Map a request layer to ``[0, num_layers]``; reject anything else."""
    if type(layer) is not int:
        raise FeatureCaptureError(f"layer must be an int, got {type(layer).__name__}")
    resolved = layer + num_layers + 1 if layer < 0 else layer
    if not 0 <= resolved <= num_layers:
        raise FeatureCaptureError(
            f"layer {layer} is outside [0, {num_layers}] "
            f"(hidden_states has {num_layers + 1} entries; -1 means {num_layers})"
        )
    return resolved


def _resolve_explicit(index: Any, prompt_len: int) -> Tuple[int, ...]:
    if not isinstance(index, (list, tuple)) or not index:
        raise FeatureCaptureError("positions.index must be a non-empty list of ints")
    resolved = []
    for position in index:
        if type(position) is not int:
            raise FeatureCaptureError(
                f"position must be an int, got {type(position).__name__}"
            )
        absolute = position + prompt_len if position < 0 else position
        if not 0 <= absolute < prompt_len:
            raise FeatureCaptureError(
                f"position {position} is outside the {prompt_len}-token prompt"
            )
        resolved.append(absolute)
    # Preserve caller order but collapse duplicates: the same row must not be
    # written twice into one output buffer.
    seen, ordered = set(), []
    for position in resolved:
        if position not in seen:
            seen.add(position)
            ordered.append(position)
    return tuple(ordered)


def _resolve_suffix(
    suffix: Any, prompt_ids: Sequence[int]
) -> Tuple[int, int]:
    """Return ``(end, occurrences)`` for the last occurrence of ``suffix``.

    ``end`` is exclusive: the captured position is ``end - 1``, i.e. the last
    token of the suffix, which is the SafeGauge convention.
    """
    if not isinstance(suffix, (list, tuple)) or not suffix:
        raise FeatureCaptureError(
            "positions.suffix_token_ids must be a non-empty list of token ids"
        )
    if any(type(token) is not int for token in suffix):
        raise FeatureCaptureError(
            "positions.suffix_token_ids must contain only integer token ids"
        )
    suffix = tuple(suffix)
    width = len(suffix)
    occurrences, last_end = 0, -1
    for start in range(0, len(prompt_ids) - width + 1):
        if tuple(prompt_ids[start : start + width]) == suffix:
            occurrences += 1
            last_end = start + width
    if last_end < 0:
        raise FeatureCaptureSuffixNotFound(
            f"suffix of {width} token(s) does not occur in the {len(prompt_ids)}-token "
            "prompt; refusing to fall back to the last prompt token"
        )
    return last_end, occurrences


def validate(
    raw: Any,
    *,
    num_layers: int,
    hidden_size: int,
    prompt_ids: Optional[Sequence[int]] = None,
    suffix_token_ids: Optional[Sequence[int]] = None,
    max_targets: int = DEFAULT_MAX_TARGETS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> FeatureRequestSpec:
    """Validate a raw ``feature_capture`` request and resolve its positions.

    Args:
        raw: the request-side mapping, e.g.
            ``{"layers": [-1], "positions": {"mode": "last"}}``.
        num_layers: decoder layer count ``N`` from the model config.
        hidden_size: model hidden width, used only for the byte budget.
        prompt_ids: tokenized prompt. Required because validation resolves every
            requested position to an absolute prompt index.
        suffix_token_ids: pre-tokenized prompt suffix, filling in
            ``positions.index`` when the caller passed text instead of ids.
        max_targets: upper bound on ``len(layers)``.
        max_bytes: raw-output budget, estimated at FP32 per element.

    Returns:
        A ``FeatureRequestSpec`` whose ``positions`` are absolute indices.

    Raises:
        FeatureCaptureError: on any unsupported or out-of-range request.
    """
    if not isinstance(raw, Mapping):
        raise FeatureCaptureError("feature_capture must be a mapping")
    _reject_unknown(raw, ("layers", "site", "positions", "encoding"), "feature_capture")

    site = raw.get("site", "block_output")
    if site not in SUPPORTED_SITES:
        raise FeatureCaptureError(
            f"site {site!r} is not implemented; supported: {', '.join(SUPPORTED_SITES)}"
        )

    layers = raw.get("layers")
    if not isinstance(layers, (list, tuple)) or not layers:
        raise FeatureCaptureError("layers must be a non-empty list of ints")
    if len(layers) > max_targets:
        raise FeatureCaptureError(
            f"at most {max_targets} layers per request, got {len(layers)}"
        )
    resolved_layers: list[int] = []
    for layer in layers:
        resolved = _resolve_layer(layer, num_layers)
        if resolved in resolved_layers:
            raise FeatureCaptureError(f"duplicate layer {layer} (resolves to {resolved})")
        resolved_layers.append(resolved)

    encoding = raw.get("encoding", "base64")
    if encoding not in SUPPORTED_ENCODINGS:
        raise FeatureCaptureError(
            f"encoding must be one of {', '.join(SUPPORTED_ENCODINGS)}"
        )

    position_spec = raw.get("positions", {"mode": "last"})
    if not isinstance(position_spec, Mapping):
        raise FeatureCaptureError("positions must be a mapping with a mode")
    mode = position_spec.get("mode", "last")
    if mode not in POSITION_MODES:
        raise FeatureCaptureError(
            f"positions.mode must be one of {', '.join(POSITION_MODES)}"
        )

    if prompt_ids is None:
        raise FeatureCaptureError(
            f"positions.mode {mode!r} needs the tokenized prompt, "
            "which is not available yet"
        )
    if not prompt_ids:
        raise FeatureCaptureError("feature_capture requires a non-empty tokenized prompt")

    if mode == "last":
        _reject_unknown(position_spec, ("mode",), "positions")
        positions = (len(prompt_ids) - 1,)
    elif mode == "all":
        _reject_unknown(position_spec, ("mode",), "positions")
        positions = tuple(range(len(prompt_ids)))
    elif mode == "explicit":
        _reject_unknown(position_spec, ("mode", "index"), "positions")
        positions = _resolve_explicit(position_spec.get("index"), len(prompt_ids))
    else:  # suffix_last
        _reject_unknown(
            position_spec, ("mode", "suffix_token_ids", "suffix_text"), "positions"
        )
        suffix = position_spec.get("suffix_token_ids")
        if suffix is None:
            # The tokenizer encodes ``suffix_text`` and passes the ids through
            # ``suffix_token_ids``; nothing here may call a tokenizer.
            if position_spec.get("suffix_text") is not None and suffix_token_ids is None:
                raise FeatureCaptureError(
                    "positions.suffix_text must be tokenized by the caller before "
                    "validation; pass the ids as suffix_token_ids"
                )
            suffix = suffix_token_ids
            if suffix is None:
                raise FeatureCaptureError(
                    "positions.mode 'suffix_last' needs positions.suffix_token_ids "
                    "or positions.suffix_text"
                )
        end, occurrences = _resolve_suffix(suffix, prompt_ids)
        positions = (end - 1,)
        _check_budget(len(resolved_layers), len(positions), hidden_size, max_bytes)
        return FeatureRequestSpec(
            targets=tuple(FeatureTarget(layer, site) for layer in resolved_layers),
            positions=positions,
            position_mode=mode,
            encoding=encoding,
            suffix_occurrences=occurrences,
        )

    _check_budget(len(resolved_layers), len(positions), hidden_size, max_bytes)
    return FeatureRequestSpec(
        targets=tuple(FeatureTarget(layer, site) for layer in resolved_layers),
        positions=positions,
        position_mode=mode,
        encoding=encoding,
    )


def _check_budget(captures: int, positions: int, hidden_size: int, max_bytes: int) -> None:
    # FP32 per element: deliberately pessimistic so a BF16 model cannot pass a
    # request that a float32 activations path would exceed.
    needed = captures * positions * hidden_size * 4
    if needed > max_bytes:
        raise FeatureCaptureError(
            f"capture would need {needed} raw bytes (FP32-estimated), "
            f"budget is {max_bytes}; select fewer layers or positions"
        )
