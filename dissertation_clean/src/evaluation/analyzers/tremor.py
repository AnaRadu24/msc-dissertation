"""Clinical-tremor analyser (Layer 2) — the R4 intention-tremor crescendo. Pure numpy/scipy.

Intention tremor is defined clinically by a CRESCENDO: a 3-8 Hz lateral oscillation whose
amplitude GROWS as the hand approaches the target. The discriminating measurement is therefore
to bin the reach by DISTANCE-TO-TARGET (not time) and ask whether the lateral-oscillation RMS
rises on approach.

    lateral deviation from the straight reach line
      -> detrend (Savitzky-Golay) to isolate the oscillation
      -> RMS within distance-to-target bins
      -> intention index = RMS(near) / RMS(far)   (>1 = crescendo = intention-like)

This is also the mechanism-discriminating test: signal-dependent motor noise scales with |command|,
smallest near the target, so it gives NO crescendo (index < 1) — the opposite of delay / forward-model
corruption. The full amplitude-vs-distance curve is the figure; the index is the scalar.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..engine.types import RolloutResult

_M_TO_CM = 100.0


def _detrend_savgol(x: np.ndarray, dt: float, smooth_ms: float) -> np.ndarray:
    """Residual of x [T, B] after Savitzky-Golay smoothing along time — isolates the oscillation
    from the slow approach trend. Falls back to mean-removal if the window would exceed the series."""
    from scipy.signal import savgol_filter
    T = x.shape[0]
    win = int(round(smooth_ms / (dt * 1000.0)))
    win = max(5, win | 1)                                          # force odd, >= 5
    if win >= T:
        return x - x.mean(axis=0, keepdims=True)
    return x - savgol_filter(x, win, polyorder=3, axis=0)

def _detrend_highpass(x: np.ndarray, dt: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Remove sub-``cutoff_hz`` content (the slow approach trend) while KEEPING the tremor band.

    A fixed-window Savitzky-Golay detrend can absorb a slow ~0.8 Hz oscillation as 'trend' and
    subtract it out; a Butterworth high-pass at ``cutoff_hz`` preserves anything above it. Falls back
    to mean-removal if the series is too short to filter. ``x`` [T, B] -> [T, B]."""
    from scipy.signal import butter, filtfilt
    T = x.shape[0]
    nyq = 0.5 / dt
    if cutoff_hz >= nyq or T < 12:
        return x - x.mean(axis=0, keepdims=True)
    b, a = butter(2, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, x, axis=0)


def intention_tremor_profile(
    trajectories,
    targets,
    dt,
    *,
    dist_edges_cm=(10, 8, 6, 4, 3, 2, 1.5, 1, 0.5, 0),
    detrend="highpass",          # 'highpass' keeps the tremor band; 'savgol' legacy
    smooth_ms=120.0,             # savgol only
    highpass_hz=0.5,             # highpass only
    near_cm=2.0,
    far_cm=6.0,
):
    """Amplitude-vs-distance profile of the lateral oscillation + the intention index.
    Self-contained: computes the signed lateral deviation and distance-to-target inline, so it does
    not depend on any module-level geometry helper.

    Returns per-trial arrays:
        bin_centers_cm [nbins], rms_by_bin_cm [nbins,B], intention_index [B] (>1 = crescendo).
    """
    traj = np.asarray(trajectories, dtype=float)      # [T+1, B, 2] m
    targets = np.asarray(targets, dtype=float)        # [B, 2] m

    # --- inline reach geometry: signed lateral deviation + distance-to-target ---
    start = traj[0]                                    # [B, 2]
    axis = targets - start                            # [B, 2]
    u = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9, None)  # unit reach axis
    rel = traj - start[None]                          # [T+1, B, 2]
    lateral = u[None, :, 0] * rel[..., 1] - u[None, :, 1] * rel[..., 0]   # signed perpendicular [T+1,B] m
    r_cm = np.linalg.norm(traj - targets[None], axis=-1) * 100.0          # distance to target [T+1,B] cm

    # --- detrend the lateral signal to isolate the oscillation ---
    lat_cm = lateral * 100.0
    if detrend == "highpass":
        resid_cm = _detrend_highpass(lat_cm, dt, highpass_hz)
    else:
        from scipy.signal import savgol_filter
        T = lat_cm.shape[0]
        win = max(5, int(round(smooth_ms / (dt * 1000.0))) | 1)
        if win >= T:
            win = (T - 1) if (T - 1) % 2 else (T - 2)
        resid_cm = (lat_cm - lat_cm.mean(0, keepdims=True)) if win < 5 \
            else lat_cm - savgol_filter(lat_cm, win, 3, axis=0)

    B = traj.shape[1]
    edges = np.asarray(dist_edges_cm, float)          # descending, e.g. 10 -> 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    rms = np.full((len(centers), B), np.nan)
    for i in range(len(centers)):
        hi, lo = edges[i], edges[i + 1]
        for b in range(B):
            m = (r_cm[:, b] < hi) & (r_cm[:, b] >= lo)
            if m.sum() >= 2:
                rms[i, b] = np.sqrt(np.mean(resid_cm[m, b] ** 2))

    idx = np.full(B, np.nan)
    for b in range(B):
        near, far = r_cm[:, b] < near_cm, r_cm[:, b] > far_cm
        if near.sum() >= 2 and far.sum() >= 2:
            rn = np.sqrt(np.mean(resid_cm[near, b] ** 2))
            rf = np.sqrt(np.mean(resid_cm[far, b] ** 2))
            idx[b] = rn / rf if rf > 1e-9 else np.nan

    return {"bin_centers_cm": centers, "rms_by_bin_cm": rms, "intention_index": idx}


def tremor_metrics(res: RolloutResult, **kw) -> Dict[str, np.ndarray]:
    """Per-trial tremor battery for the atom: just the scalar intention index (the profile
    curve is a figure input, not a stored scalar)."""
    profile = intention_tremor_profile(res.trajectories, res.targets, res.dt, **kw)
    return {"intention_index": profile["intention_index"]}