"""R4 payoff — the hypothesis × signature matrix. Runs each MS hypothesis (as a test-time manipulation
of the SAME dual model) through a common battery of motor signatures, so the model's power to
DISCRIMINATE hypotheses is shown in one table. Pure orchestration + a pandas frame; figure is separate.

Signatures per hypothesis (all in-distribution, 1 s rollout, hold 0.6-1.0 s):
    terminal_error_cm   reach competence retained?
    hold_rms_cms        postural hold instability
    crescendo_index     kinetic (intention) vs postural; >1 = crescendo
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .analyzers.reach import terminal_error_cm
from .analyzers.tremor import intention_tremor_profile
from .catalogue import Catalogue, ModelRecord
from .engine import obs_layout, rollout
from .engine.types import IDENTITY, Intervention
from .manipulators.delay import delay
from .manipulators.noise import noise
from .paths import Paths
from .settings import DEFAULT, Settings

_HOLD_LO, _HOLD_HI = 0.6, 1.0


def _hold_rms(res):
    traj = np.asarray(res.trajectories)
    speed = np.linalg.norm(np.diff(traj, axis=0) / res.dt, axis=-1) * 100.0
    t = np.arange(speed.shape[0]) * res.dt
    m = (t >= _HOLD_LO) & (t < _HOLD_HI)
    return float(np.sqrt((speed[m] ** 2).mean(0)).mean()) if m.any() else np.nan


def _signatures(res) -> Dict[str, float]:
    prof = intention_tremor_profile(res.trajectories, res.targets, res.dt, detrend="highpass")
    return {
        "terminal_error_cm": float(np.nanmean(terminal_error_cm(res))),
        "hold_rms_cms": _hold_rms(res),
        "crescendo_index": float(np.nanmean(prof["intention_index"])),
    }


def hypothesis_signature_matrix(
    cat: Catalogue, paths: Paths, *,
    delta_ms: float = 60.0, sigma: float = 0.3,
    settings: Settings = DEFAULT,
) -> pd.DataFrame:
    """Run each MS hypothesis through the signature battery, averaged across the canonical dual seeds
    (and, for the high-gain hypothesis, its own trained variant). Returns a tidy DataFrame:
    rows = (hypothesis, seed), columns = signatures. Aggregate to mean±SD downstream."""
    dual = cat.canonical("dual")
    rows = []

    def run(rec: ModelRecord, iv: Intervention, task="reach"):
        return rollout(rec, paths, task=task, duration_s=1.0, batch_size=128,
                       intervention=iv, seed=settings.rollout_seed)

    for rec in dual.records:
        layout = obs_layout(rec)
        bp, bv = rec.config.proprioception_delay, rec.config.vision_delay
        hyps = {
            "Healthy (baseline)": IDENTITY,
            "Pure delay": delay(delta_ms, mode="additive_both", base_prop_ms=bp, base_vision_ms=bv),
            "Forward-model breakdown": noise("predictor_output", sigma, layout=layout,
                                             architecture="dual", seed=settings.rollout_seed),
            "Signal-dependent noise": noise("command", sigma, layout=layout, signal_dependent=True,
                                            architecture="dual", seed=settings.rollout_seed),
        }
        for name, iv in hyps.items():
            sig = _signatures(run(rec, iv))
            rows.append({"hypothesis": name, "seed": rec.seed, **sig})

    # High-gain compensation = the separately-TRAINED velocity-penalty variant, at baseline (no test-time iv)
    hi_gain = cat.select(architecture="dual")   # find the velocity_penalty variant by label
    for rec in [r for r in hi_gain if "velocity_penalty" in r.label]:
        sig = _signatures(run(rec, IDENTITY))
        rows.append({"hypothesis": "High-gain (trained variant)", "seed": rec.seed, **sig})

    return pd.DataFrame(rows)


def aggregate_signatures(df: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Collapse across seeds: mean ± SD per (hypothesis, signature)."""
    sig_cols = ["terminal_error_cm", "hold_rms_cms", "crescendo_index"]
    g = df.groupby("hypothesis")[sig_cols]
    m = g.mean(); s = g.std(ddof=ddof)
    out = m.copy()
    for c in sig_cols:
        out[c] = [f"{mv:.2f} ± {sv:.2f}" for mv, sv in zip(m[c], s[c])]
    # keep a numeric copy for the heatmap colouring
    return out, m