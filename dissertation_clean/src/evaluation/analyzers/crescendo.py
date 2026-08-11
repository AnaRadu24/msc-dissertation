"""Reach-phase intention-tremor crescendo analysers — PURE numpy. Deceleration-phase amplitude.

Intention tremor is defined as lateral-oscillation amplitude GROWING during the DECELERATION phase
(peak velocity -> target arrival). We bin by TIME within that phase, NOT by distance-to-target,
because a tremulous reach overshoots/undershoots — the same distance is visited many times, so
distance bins conflate different moments. Time is monotonic through deceleration, so it is the correct
axis and it matches the clinical definition (growth on final approach).

amplitude = peak-to-peak lateral excursion in a sliding time window; growth ratio = late/early.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _lateral_cm(traj, targets):
    """Signed lateral (perpendicular-to-reach) deviation, cm. [T+1,B]."""
    start = traj[0]; axis = targets - start
    u = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9, None)
    rel = traj - start[None]
    return (u[None, :, 0] * rel[..., 1] - u[None, :, 1] * rel[..., 0]) * 100.0

def _hold_onset_index(speed_b, i_peak, dt, *, frac=0.1, sustain_ms=100.0, floor_cms=3.0):
    """First index after peak velocity where speed stays below max(frac*peak, floor) for
    sustain_ms. Dynamics-based (velocity stabilises), not a fixed time/distance. Returns the
    episode end if the arm never settles (a sustained tremor has NO hold onset — which is the point)."""
    peak = speed_b[i_peak]
    thr = max(frac * peak, floor_cms)
    sustain = max(1, int(round(sustain_ms / (dt * 1000.0))))
    below = speed_b < thr
    for i in range(i_peak, len(speed_b) - sustain):
        if below[i:i + sustain].all():
            return i
    return len(speed_b) - 1                                    # never settles -> whole episode

def _phase_end_index(speed_b, r_cm_b, i_peak, dt, *, arrival_cm=2.0, frac=0.1, floor_cms=3.0):
    """End of the deceleration→settle phase: the FIRST of (a) first entry within arrival_cm of target,
    or (b) velocity dropping below max(frac*peak, floor). For a clean reach this is arrival; for a
    tremulous reach that overshoots, it's the first near-target approach — so the window captures the
    APPROACH crescendo and stops before the post-arrival hold/limit-cycle dilutes it. If neither
    triggers (rare), fall back to minimum-distance time."""
    peak = speed_b[i_peak]
    thr = max(frac * peak, floor_cms)
    near = np.where(r_cm_b[i_peak:] < arrival_cm)[0]
    slow = np.where(speed_b[i_peak:] < thr)[0]
    cand = [i_peak + int(x[0]) for x in (near, slow) if x.size]
    return min(cand) if cand else int(np.argmin(r_cm_b))


def deceleration_crescendo(trajectories, targets, dt, *, n_bins=6, window_ms=150.0,
                           hold_frac=0.1, sustain_ms=100.0):
    """Lateral swing amplitude from PEAK VELOCITY to HOLD-ONSET (velocity-stabilisation, not a fixed
    arrival distance — so a never-settling tremor keeps its full window). Binned by normalised time.
    Per trial: amp_by_bin [n_bins,B], growth_ratio [B] (last/first bin), plus the phase endpoints."""
    traj = np.asarray(trajectories, float); targets = np.asarray(targets, float)
    T, B = traj.shape[0], traj.shape[1]
    lateral = _lateral_cm(traj, targets)
    speed = np.concatenate([np.zeros((1, B)),
                            np.linalg.norm(np.diff(traj, axis=0), axis=-1) / dt], axis=0) * 100.0
    win = max(1, int(round(window_ms / (dt * 1000.0))))
    centers = (np.arange(n_bins) + 0.5) / n_bins
    amp = np.full((n_bins, B), np.nan); growth = np.full(B, np.nan)
    i_peaks = np.zeros(B, int); i_holds = np.zeros(B, int)

    for b in range(B):
        i_peak = int(np.argmax(speed[:, b]))
        r_cm_b = np.linalg.norm(traj[:, b] - targets[b], axis=-1) * 100.0
        i_end = _phase_end_index(speed[:, b], r_cm_b, i_peak, dt)
        i_peaks[b], i_holds[b] = i_peak, i_end   # (keep the name i_hold in the return for the lines)
        if i_end - i_peak < n_bins:
            continue
        idx = np.linspace(i_peak, i_end, n_bins + 1).astype(int)
        for k in range(n_bins):
            seg = lateral[idx[k]:min(max(idx[k] + win, idx[k + 1]), T), b]
            if seg.size >= 2:
                amp[k, b] = seg.max() - seg.min()
        if np.isfinite(amp[0, b]) and np.isfinite(amp[-1, b]) and amp[0, b] > 1e-6:
            growth[b] = amp[-1, b] / amp[0, b]
    return {"time_norm": centers, "amp_by_bin_cm": amp, "growth_ratio": growth,
            "i_peak": i_peaks, "i_hold": i_holds}

def deceleration_crescendo_old(trajectories, targets, dt, *, n_bins=6, window_ms=150.0,
                           arrival_cm=1.0):
    """Lateral swing amplitude across the DECELERATION phase (peak-velocity -> first arrival), binned
    by normalised time (0=peak vel, 1=arrival). Per trial:
        amp_by_bin [n_bins, B]  peak-to-peak lateral swing (cm) in each time-bin of deceleration
        growth_ratio [B]        amp(last bin) / amp(first bin) — clinical 'amplitude grows on approach'
    Returns bin centres (normalised time) too. Overshoot-robust (time axis is monotonic)."""
    traj = np.asarray(trajectories, float); targets = np.asarray(targets, float)
    T, B = traj.shape[0], traj.shape[1]
    lateral = _lateral_cm(traj, targets)
    r_cm = np.linalg.norm(traj - targets[None], axis=-1) * 100.0
    speed = np.concatenate([np.zeros((1, B)),
                            np.linalg.norm(np.diff(traj, axis=0), axis=-1) / dt], axis=0) * 100.0
    win = max(1, int(round(window_ms / (dt * 1000.0))))
    centers = (np.arange(n_bins) + 0.5) / n_bins
    amp = np.full((n_bins, B), np.nan)
    growth = np.full(B, np.nan)

    for b in range(B):
        i_peak = int(np.argmax(speed[:, b]))                      # peak velocity
        below = np.where(r_cm[i_peak:, b] < arrival_cm)[0]
        i_arr = (i_peak + int(below[0])) if below.size else int(np.argmin(r_cm[:, b]))
        if i_arr - i_peak < n_bins:                               # too short to bin
            continue
        idx = np.linspace(i_peak, i_arr, n_bins + 1).astype(int)
        for k in range(n_bins):
            lo, hi = idx[k], max(idx[k] + win, idx[k + 1])
            seg = lateral[lo:min(hi, T), b]
            if seg.size >= 2:
                amp[k, b] = seg.max() - seg.min()
        if np.isfinite(amp[0, b]) and np.isfinite(amp[-1, b]) and amp[0, b] > 1e-6:
            growth[b] = amp[-1, b] / amp[0, b]
    return {"time_norm": centers, "amp_by_bin_cm": amp, "growth_ratio": growth}