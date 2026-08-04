"""Reach-quality analysers (Layer 2) — R1 competence. Pure numpy: no matplotlib, no torch.

Every function takes a RolloutResult and returns PER-TRIAL arrays [B]; the store collapses
those to mean/std across trials, and cross-seed aggregation happens later. Distances are
returned in centimetres, speeds in the stated unit.

REACH ACCURACY is the distance to target at the FINAL timestep of the rollout. When the
rollout is run under the 'competence' profile (trained horizon) this is the in-distribution
reach-and-hold error we defined as the R1 number — no averaging window, no speed heuristic.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..engine.types import RolloutResult

_M_TO_CM = 100.0


def _distance_to_target_cm(res: RolloutResult) -> np.ndarray:
    """[T+1, B] Euclidean fingertip-to-target distance over time, in cm."""
    traj = np.asarray(res.trajectories, dtype=float)       # [T+1, B, 2] (m)
    tgt = np.asarray(res.targets, dtype=float)[None]       # [1, B, 2]
    return np.linalg.norm(traj - tgt, axis=-1) * _M_TO_CM


def _speed_cms(res: RolloutResult) -> np.ndarray:
    """[T, B] fingertip speed over time, in cm/s (finite difference of position)."""
    traj = np.asarray(res.trajectories, dtype=float)
    return np.linalg.norm(np.diff(traj, axis=0) / res.dt, axis=-1) * _M_TO_CM


def terminal_error_cm(res: RolloutResult) -> np.ndarray:
    """[B] THE R1 accuracy number: distance to target at the final timestep of the rollout.

    Under the 'competence' profile the final timestep IS the trained horizon, so this is the
    reach-and-hold error the model was trained to minimise — in-distribution, zero free
    parameters. (If you pass a longer rollout this becomes end-of-6s error, which is drift,
    not accuracy — so only read this off a competence-profile rollout.)
    """
    return _distance_to_target_cm(res)[-1]


def peak_speed_cms(res: RolloutResult) -> np.ndarray:
    """[B] peak fingertip speed — the reach vigour, used for the speed-matched control."""
    return _speed_cms(res).max(axis=0)


def path_efficiency(res: RolloutResult) -> np.ndarray:
    """[B] straight-line distance / actual path length (1.0 = perfectly straight reach)."""
    traj = np.asarray(res.trajectories, dtype=float)
    step = np.linalg.norm(np.diff(traj, axis=0), axis=-1)          # [T, B]
    actual = step.sum(axis=0)                                      # [B]
    ideal = np.linalg.norm(res.targets - traj[0], axis=-1)         # [B]
    return np.divide(ideal, actual, out=np.ones_like(actual), where=actual > 1e-9)


def muscle_effort(res: RolloutResult) -> np.ndarray:
    """[B] total muscle activation summed over muscles and time (energetic cost proxy)."""
    u = np.asarray(res.motor_commands, dtype=float)                # [T, B, 6]
    return u.sum(axis=2).sum(axis=0)


def reach_metrics(res: RolloutResult) -> Dict[str, np.ndarray]:
    """The full R1 per-trial battery. Keys are the metric names stored in the atom."""
    return {
        "terminal_error_cm": terminal_error_cm(res),
        "peak_speed_cms":    peak_speed_cms(res),
        "path_efficiency":   path_efficiency(res),
        "muscle_effort":     muscle_effort(res),
    }