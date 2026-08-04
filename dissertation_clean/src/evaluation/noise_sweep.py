"""Noise-injection sweep — R4's mechanistic dissociation. Sweeps Gaussian-noise amplitude at each
loop SITE and measures the resulting hold instability, IN-DISTRIBUTION (1 s rollout, hold window
within the trained horizon), for the same reason the delay sweep is in-distribution: the claim must
not rest on out-of-distribution extrapolation.

The point only a computational model can make: inject identical-magnitude corruption at anatomically
distinct sites (afferent-to-cerebellum, cerebellar output, descending command, raw feedback) and read
out WHICH site produces the pathology. The dissociation: state-estimation sites destabilise the hold;
the descending-command site does not (command noise scales with command magnitude, smallest near the
target, so the closed loop rejects it — Harris-Wolpert).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .catalogue import ModelRecord
from .engine import obs_layout, rollout
from .manipulators.noise import SITES, noise
from .paths import Paths
from .settings import DEFAULT, Settings

_HOLD_LO, _HOLD_HI = 0.6, 1.0        # in-distribution hold window (s) within the trained 1 s


def _hold_rms(res, lo_s=_HOLD_LO, hi_s=_HOLD_HI) -> float:
    """Mean hold-phase speed RMS (cm/s) over [lo_s, hi_s] of the rollout."""
    traj = np.asarray(res.trajectories)
    speed = np.linalg.norm(np.diff(traj, axis=0) / res.dt, axis=-1) * 100.0
    t = np.arange(speed.shape[0]) * res.dt
    m = (t >= lo_s) & (t < hi_s)
    return float(np.sqrt((speed[m] ** 2).mean(0)).mean()) if m.any() else np.nan


@dataclass
class NoiseSweepResult:
    model: str
    seed: int
    site: str
    sigma: np.ndarray            # [G] noise amplitudes
    hold_rms_cms: np.ndarray     # [G] in-distribution hold-RMS at each sigma


def noise_sweep(
    record: ModelRecord,
    paths: Paths,
    *,
    sites: Sequence[str] = SITES,
    sigmas: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
    signal_dependent_command: bool = True,   # command site uses Harris-Wolpert scaling
    settings: Settings = DEFAULT,
) -> List[NoiseSweepResult]:
    """Sweep noise amplitude at each site for one model-seed; return a curve per site.
    Predictor sites are skipped for mono models (no predictive net)."""
    layout = obs_layout(record)
    out = []
    for site in sites:
        if site in ("predictor_input", "predictor_output") and record.architecture != "dual":
            continue
        xs, ys = [], []
        for s in sigmas:
            if s == 0:
                from .engine.types import IDENTITY
                iv = IDENTITY
            else:
                iv = noise(site, s, layout=layout,
                           signal_dependent=(signal_dependent_command and site == "command"),
                           architecture=record.architecture, seed=settings.rollout_seed)
            res = rollout(record, paths, task="reach", duration_s=1.0, batch_size=128,
                          intervention=iv, seed=settings.rollout_seed)
            xs.append(float(s)); ys.append(_hold_rms(res))
        out.append(NoiseSweepResult(record.label, record.seed, site,
                                    np.array(xs), np.array(ys)))
    return out

def noise_sweep_group(group, paths, *, sites=SITES,
                      sigmas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4),
                      signal_dependent_command=True, settings=DEFAULT):
    """Run noise_sweep for every seed of a group. Returns {site: {'sigma':[G], 'mean':[G],
    'sd':[G], 'n':[G]}} aggregated ACROSS SEEDS (mean/SD of per-seed hold-RMS at each sigma)."""
    per_seed = {}                                   # site -> list of [G] arrays (one per seed)
    sig_ref = np.asarray(sigmas, float)
    for rec in group.records:
        for sw in noise_sweep(rec, paths, sites=sites, sigmas=sigmas,
                              signal_dependent_command=signal_dependent_command, settings=settings):
            per_seed.setdefault(sw.site, []).append(sw.hold_rms_cms)
    out = {}
    for site, curves in per_seed.items():
        M = np.vstack(curves)                       # [n_seeds, G]
        out[site] = {"sigma": sig_ref,
                     "mean": np.nanmean(M, axis=0),
                     "sd": np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.full(M.shape[1], np.nan),
                     "n": np.sum(~np.isnan(M), axis=0)}
    return out