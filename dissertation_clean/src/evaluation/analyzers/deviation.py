"""Trajectory-deviation analyser (Layer 2) — how much an intervention reshaped the PATH.

Compares two PAIRED rollouts (same seed => identical targets, start posture, trial order, dt and
step count) point-by-point in time. More sensitive than endpoint error: a closed-loop controller
can reshape its trajectory (curvature, timing, online correction) yet still land on target, leaving
terminal error ~unchanged. A near-zero deviation across the whole path is therefore much stronger
evidence of OPEN-loop behaviour than a near-zero endpoint error alone.

Pure numpy. Because it needs TWO rollouts it is not a plain RolloutResult->metrics analyser; the
orchestrator calls it with (intervened, baseline) when a spec declares a paired comparison.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..engine.types import RolloutResult

_M_TO_CM = 100.0


def trajectory_deviation(result: RolloutResult, reference: RolloutResult) -> Dict[str, np.ndarray]:
    """Per-trial [B] deviation (cm) between ``result`` and a paired ``reference`` rollout.

    Both must share seed, batch and step count (raised otherwise), so points correspond one-to-one
    in time and a per-timestep Euclidean distance is exactly the right comparison — no time-warping.
    Returns mean / max / RMS of the point-wise distance per trial.
    """
    a = np.asarray(result.trajectories, dtype=float)
    b = np.asarray(reference.trajectories, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"trajectory_deviation: shape mismatch {a.shape} vs {b.shape}; the two "
                         "rollouts must share seed, batch size and step count.")
    d = np.linalg.norm(a - b, axis=-1) * _M_TO_CM        # [T+1, B] point-wise distance
    return {
        "traj_dev_mean_cm": d.mean(axis=0),
        "traj_dev_max_cm":  d.max(axis=0),
        "traj_dev_rms_cm":  np.sqrt((d ** 2).mean(axis=0)),
    }