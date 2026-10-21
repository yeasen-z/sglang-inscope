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
"""Model-facing callback bridge for request-scoped feature capture.

The bridge deliberately owns no process-global state. A forward either carries
a recorder or passes ``None``; disabled and ordinary requests therefore take a
single null check while existing auxiliary hidden-state callbacks retain full
ownership of their model behavior.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

CaptureHook = Callable[..., None]


class FeatureCaptureRecorder(Protocol):
    """The only recorder operation visible to model adapters."""

    def capture_hook(self, *, layer: int) -> Optional[CaptureHook]: ...


class _CaptureAccumulator:
    """Adapt a layer-boundary callback to the accumulator-style interface."""

    def __init__(self, hook: CaptureHook):
        self._hook = hook

    def capture(self, value: Any, *, owned: bool = False) -> None:
        self._hook(value, owned=owned)


def layer_capture(
    *,
    layer: int,
    recorder: Optional[FeatureCaptureRecorder],
    auxiliary_hook: Optional[CaptureHook] = None,
) -> Optional[CaptureHook]:
    """Return one callback that preserves both capture consumers."""
    feature_hook = recorder.capture_hook(layer=layer) if recorder is not None else None
    return _combine(feature_hook=feature_hook, auxiliary_hook=auxiliary_hook)


def layer_capture_accumulator(
    *,
    layer: int,
    recorder: Optional[FeatureCaptureRecorder],
    auxiliary_accumulator: Optional[Any] = None,
) -> Optional[Any]:
    """Return a collector for models whose boundary expects ``.capture()``."""
    hook = layer_capture(
        layer=layer,
        recorder=recorder,
        auxiliary_hook=(
            auxiliary_accumulator.capture
            if auxiliary_accumulator is not None
            else None
        ),
    )
    return _CaptureAccumulator(hook) if hook is not None else None


def final_capture(
    *, num_layers: int, recorder: Optional[FeatureCaptureRecorder]
) -> Optional[CaptureHook]:
    """Return the callback for block boundary ``N`` (pre-final-norm)."""
    if recorder is None:
        return None
    return _combine(
        feature_hook=recorder.capture_hook(layer=num_layers), auxiliary_hook=None
    )


def _combine(
    *, feature_hook: Optional[CaptureHook], auxiliary_hook: Optional[CaptureHook]
) -> Optional[CaptureHook]:
    if feature_hook is None:
        return auxiliary_hook

    def combined(value: Any, *, owned: bool = False) -> None:
        # Feature capture never trusts ownership: its recorder copies selected
        # rows before later layers can reuse the producer's storage.
        feature_hook(value, owned=False)
        if auxiliary_hook is not None:
            auxiliary_hook(value, owned=owned)

    return combined
