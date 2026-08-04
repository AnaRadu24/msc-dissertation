"""Noise-injection manipulator (Layer 3) — Gaussian noise at a loop site at test time.

Models cerebellar dysfunction as unreliable internal signals (a corrupted forward model) rather
than pure delay. Sites: predictor_input / predictor_output (dual only), command (with optional
Harris-Wolpert signal-dependence), feedback (leaves the goal clean). Each is a reproducible,
seeded hook. Pure factory: returns an Intervention. No torch-load, no rollout.
"""
from __future__ import annotations

from typing import Optional

import torch as th

from ..engine.types import Intervention, ObsLayout

SITES = ("predictor_input", "predictor_output", "command", "feedback")
_DUAL_ONLY = ("predictor_input", "predictor_output")
_HOOK = {  # which Intervention field each site drives
    "predictor_input":  "pred_input_transform",
    "predictor_output": "estimate_transform",
    "command":          "command_transform",
    "feedback":         "obs_transform",
}


class _Noise:
    """Add seeded Gaussian noise. ``slice_`` restricts it to columns (leave the goal clean for
    'feedback'); ``signal_dependent`` scales SD by |x| element-wise (Harris-Wolpert)."""
    def __init__(self, sigma: float, *, signal_dependent: bool, slice_: Optional[slice], seed: int):
        self.sigma, self.signal_dependent, self.slice_ = float(sigma), signal_dependent, slice_
        self._gen = th.Generator(device="cpu"); self._gen.manual_seed(seed)

    def _noise(self, seg: th.Tensor) -> th.Tensor:
        n = th.randn(seg.shape, generator=self._gen).to(seg.device)
        return n * (self.sigma * (seg.abs() if self.signal_dependent else 1.0))

    def __call__(self, x: th.Tensor) -> th.Tensor:
        if self.sigma <= 0:
            return x
        if self.slice_ is None:
            return x + self._noise(x)
        out = x.clone()
        out[:, self.slice_] = out[:, self.slice_] + self._noise(out[:, self.slice_])
        return out


def noise(site: str, sigma: float, *, layout: Optional[ObsLayout] = None,
          signal_dependent: bool = False, architecture: str = "dual", seed: int = 0) -> Intervention:
    """Inject Gaussian noise of SD ``sigma`` at ``site``. 'feedback' needs ``layout`` (to leave the
    goal channel clean). Predictor sites require a dual model. ``signal_dependent`` (command site)
    gives Harris-Wolpert scaling. ``seed`` makes the noise reproducible."""
    if site not in SITES:
        raise ValueError(f"site must be one of {SITES}, got {site!r}")
    if site in _DUAL_ONLY and architecture != "dual":
        raise ValueError(f"site {site!r} requires a dual model (a mono has no predictive net).")
    slice_ = None
    if site == "feedback":
        if layout is None:
            raise ValueError("site 'feedback' needs a layout (to leave the goal channel clean).")
        slice_ = slice(layout.goal, None)     # everything after the goal = vision + prop
    hook = _Noise(sigma, signal_dependent=signal_dependent, slice_=slice_, seed=seed)
    tag = f"noise_{site}{'_sigdep' if signal_dependent else ''}_s{sigma:g}"
    return Intervention(
        label=tag,
        description=f"Gaussian noise σ={sigma:g} at {site}"
                    + (" (signal-dependent)" if signal_dependent else ""),
        **{_HOOK[site]: hook},
    )