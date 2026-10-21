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
"""Map a forward batch onto the rows each request wants captured.

Pure arithmetic over plain Python values: no torch import, so the row-mapping
rules are unit-testable without a GPU. Row indices returned here are positions
in the flattened token axis of one forward (``extend`` chunk or ``decode``
step), which is exactly what ``torch.index_select`` needs at the capture site.

The rule that matters most: the running offset advances over *every* request in
the batch, including ones that capture nothing. A batch of requests A/B/C with
5/4/3 extend rows where only A and C capture their last token must read rows 4
and 11 -- not 4 and 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module torch-free
    from sglang.srt.feature_capture.spec import FeatureRequestSpec


@dataclass(frozen=True)
class FeaturePlanEntry:
    """One request's contribution to this forward's capture."""

    rid: str
    layers: Tuple[int, ...]
    row_start: int
    """First row of this request inside the forward's flattened token axis."""
    position_start: int
    """Absolute token position of ``row_start``."""
    positions: Tuple[int, ...]
    """Absolute positions captured this forward (already clipped to the rows)."""


@dataclass(frozen=True)
class SlotPlan:
    """One capture slot (a layer) and where its rows go in the output buffer."""

    layer: int
    rows: Tuple[int, ...]
    """Source rows in the forward's token axis, aligned with ``segments``."""
    segments: Tuple[Tuple[str, Tuple[int, ...]], ...]
    """``(rid, absolute positions)`` per contiguous run of ``rows``."""
    out_start: int
    """First row of this slot's output buffer."""

    @property
    def out_stop(self) -> int:
        return self.out_start + len(self.rows)


def build_entries(
    *,
    rids: Sequence[str],
    specs: Sequence[Optional["FeatureRequestSpec"]],
    extend_lens: Sequence[int],
    seq_lens: Sequence[int],
    prefix_lens: Optional[Sequence[int]] = None,
) -> Tuple[FeaturePlanEntry, ...]:
    """Clip each request's wanted positions to the tokens this forward sees.

    Args:
        rids: request ids, in the batch's row order.
        specs: resolved capture specs aligned with ``rids``; ``None`` = no capture.
        extend_lens: new tokens this forward computes per request.
        seq_lens: total sequence length after this forward, per request.
        prefix_lens: length before this forward, per request. Prefer passing
            ``ForwardBatch.extend_prefix_lens_cpu`` when it is populated: it is
            the quantity we actually want, and using it avoids relying on
            ``seq_lens == prefix_lens + extend_lens`` holding for every mode.
            Defaults to ``seq_lens - extend_lens``.

    Returns:
        One entry per request that still wants at least one row, in batch order.
        Empty tuple when nothing in this batch captures.
    """
    count = len(rids)
    if len(extend_lens) != count or len(seq_lens) != count:
        raise ValueError(
            f"packed batch metadata is inconsistent: {count} rids, "
            f"{len(extend_lens)} extend_lens, {len(seq_lens)} seq_lens"
        )
    if len(specs) != count:
        raise ValueError(f"expected {count} capture specs, got {len(specs)}")
    if prefix_lens is not None and len(prefix_lens) != count:
        raise ValueError(
            f"expected {count} prefix_lens, got {len(prefix_lens)}"
        )

    entries: List[FeaturePlanEntry] = []
    offset = 0
    for index, rid in enumerate(rids):
        rows = int(extend_lens[index])
        spec = specs[index]
        if spec is not None:
            position_start = (
                int(prefix_lens[index])
                if prefix_lens is not None
                else int(seq_lens[index]) - rows
            )
            wanted = tuple(
                position
                for position in spec.positions
                if position_start <= position < position_start + rows
            )
            if wanted:
                entries.append(
                    FeaturePlanEntry(
                        rid=rid,
                        layers=spec.layers,
                        row_start=offset,
                        position_start=position_start,
                        positions=wanted,
                    )
                )
        # Advance regardless of whether this request captured anything.
        offset += rows
    return tuple(entries)


def group_by_slot(
    entries: Sequence[FeaturePlanEntry],
) -> Tuple[Dict[int, List[Tuple[FeaturePlanEntry, Tuple[int, ...]]]], int]:
    """Group capture cells by layer, preserving batch order within each layer.

    Returns:
        ``(by_layer, total_rows)`` where ``total_rows`` is the number of
        ``(layer, position)`` cells, i.e. one output row per capture.
    """
    by_layer: Dict[int, List[Tuple[FeaturePlanEntry, Tuple[int, ...]]]] = {}
    total = 0
    for entry in entries:
        for layer in entry.layers:
            by_layer.setdefault(layer, []).append((entry, entry.positions))
            total += len(entry.positions)
    return by_layer, total


def layout_slots(entries: Sequence[FeaturePlanEntry]) -> Tuple[SlotPlan, ...]:
    """Assign every capture cell a destination row in one flat output buffer.

    The buffer is ``[total_rows, hidden]``; slots occupy disjoint slices, and
    within a slot the segments keep ``(rid, position)`` attribution so the host
    side can split the tensor back into per-request chunks.
    """
    by_layer, _ = group_by_slot(entries)
    slots: List[SlotPlan] = []
    offset = 0
    for layer in sorted(by_layer):
        rows: List[int] = []
        segments: List[Tuple[str, Tuple[int, ...]]] = []
        for entry, positions in by_layer[layer]:
            segments.append((entry.rid, positions))
            rows.extend(
                entry.row_start + position - entry.position_start
                for position in positions
            )
        slots.append(
            SlotPlan(
                layer=layer,
                rows=tuple(rows),
                segments=tuple(segments),
                out_start=offset,
            )
        )
        offset += len(rows)
    return tuple(slots)


def build_slot_plan(
    *,
    rids: Sequence[str],
    specs: Sequence[Optional["FeatureRequestSpec"]],
    extend_lens: Optional[Sequence[int]],
    seq_lens: Sequence[int],
    prefix_lens: Optional[Sequence[int]] = None,
) -> Tuple[SlotPlan, ...]:
    """Build one forward's complete slot plan, or no plan for decode/idle."""
    if extend_lens is None or not any(spec is not None for spec in specs):
        return ()
    return layout_slots(
        build_entries(
            rids=rids,
            specs=specs,
            extend_lens=extend_lens,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
        )
    )
