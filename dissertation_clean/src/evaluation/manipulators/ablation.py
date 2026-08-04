"""Feedback-ablation manipulator (Layer 3) — freeze a sensory channel at test time.

Does the controller genuinely use a feedback channel, or has it collapsed to open-loop? Freeze
a channel at its t=0 value (in-distribution, but stripped of time-varying information) and see if
behaviour degrades. Big degradation => the channel is load-bearing feedback; none => open-loop
w.r.t. that channel. Channel-aware via ObsLayout, so it never touches the env or torch.

Pure factory: returns an Intervention carrying an obs_transform hook.
"""
from __future__ import annotations

from typing import Optional

import torch as th

from ..engine.types import Intervention, ObsLayout

_CHANNELS = ("vision", "prop", "both")


class _Freeze:
    """Holds a channel's observation at its FIRST value. Reproducible: the reference is
    captured on the first call (reset) and replayed thereafter."""
    def __init__(self, layout: ObsLayout, channels: tuple[str, ...]):
        slices = {"vision": layout.vision_slice, "prop": layout.prop_slice}
        self._slices = [slices[c] for c in channels]
        self._ref: Optional[th.Tensor] = None

    def __call__(self, obs: th.Tensor) -> th.Tensor:
        if self._ref is None:
            self._ref = obs.clone()               # t=0 reference
        out = obs.clone()
        for sl in self._slices:
            out[:, sl] = self._ref[:, sl]
        return out


def ablate(channel: str, layout: ObsLayout) -> Intervention:
    """Freeze ``channel`` in {'vision', 'prop', 'both'} at its t=0 value. ``layout`` comes from
    ``engine.obs_layout(record)`` — fetch once per model in the notebook and pass it in."""
    if channel not in _CHANNELS:
        raise ValueError(f"channel must be one of {_CHANNELS}, got {channel!r}")
    channels = ("vision", "prop") if channel == "both" else (channel,)
    return Intervention(
        label=f"ablate_{channel}",
        description=f"freeze {channel} feedback at t=0 (open-loop probe, weights frozen)",
        obs_transform=_Freeze(layout, channels),
    )