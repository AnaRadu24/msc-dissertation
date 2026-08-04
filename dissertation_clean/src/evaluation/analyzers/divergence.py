"""Command-divergence analyser (Layer 2) — F3, reactive vs feedforward correction. Pure numpy.

Roll matched perturbed and unperturbed trials (same seed) and compare muscle commands over time.
If the commands diverge only AFTER one feedback delay could have elapsed, the correction is
REACTIVE (feedback-clocked, closed-loop); divergence BEFORE feedback could arrive would mean
anticipatory feedforward. The measured divergence latency should equal min(prop, vision) delay —
the fastest channel that could carry the perturbation signal.

Like trajectory_deviation this needs TWO paired rollouts; the orchestrator supplies (perturbed,
unperturbed). It returns per-trial latencies plus the predicted latency for the report/figure.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..engine.types import RolloutResult


def command_divergence(
    perturbed: RolloutResult,
    unperturbed: RolloutResult,
    *,
    rel_threshold: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """First timestep at which perturbed and unperturbed muscle commands diverge, per trial.

    Divergence = the first step where the max absolute command difference exceeds
    ``rel_threshold`` times the typical command scale (so it is robust to numerical noise:
    matched trials are bit-identical until real feedback arrives, then jump many orders).

    Returns:
        divergence_latency_ms [B] : (first divergent step) x dt, in ms. NaN if never diverges.
        predicted_latency_ms  scalar : min(prop, vision) trained delay — the expected latency.
    """
    up = np.asarray(unperturbed.motor_commands, dtype=float)     # [T, B, 6]
    pt = np.asarray(perturbed.motor_commands, dtype=float)
    T = min(up.shape[0], pt.shape[0])
    up, pt = up[:T], pt[:T]
    dt_ms = perturbed.dt * 1000.0

    scale = max(np.abs(up).mean(), 1e-9)
    diff = np.abs(pt - up).max(axis=2)                           # [T, B] max over muscles
    diverged = diff > (rel_threshold * scale)                    # [T, B] boolean

    B = up.shape[1]
    latency = np.full(B, np.nan)
    for b in range(B):
        hits = np.flatnonzero(diverged[:, b])
        if hits.size:
            latency[b] = hits[0] * dt_ms
    return {"divergence_latency_ms": latency}


def predicted_divergence_ms(record_config) -> float:
    """The expected reactive latency = the faster feedback channel's trained delay."""
    return float(min(record_config.proprioception_delay, record_config.vision_delay))