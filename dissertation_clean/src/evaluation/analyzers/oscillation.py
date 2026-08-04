"""Oscillation / tremor-amplitude analysers (Layer 2) — the R2 core. Pure numpy/scipy.

Two quantities, deliberately kept SEPARATE (PROJECT_STATE §4):

  AMPLITUDE  hold-phase speed RMS (cm/s). Speed-based. This is the Δc instability trigger
             (hold-RMS >= threshold => the hold has broken into a limit cycle).
  FREQUENCY  dominant oscillation frequency (Hz). Computed from SIGNED target-centred
             coordinates, NEVER unsigned speed: an oscillation THROUGH the target rectifies
             under |.|, doubling the apparent frequency and destroying the phase. Ring-fence
             the two — do not read frequency off the speed signal.

The hold window is [hold_start, end]; hold_start defaults to the model's trained horizon
(carried on the RolloutResult), so the out-of-distribution hold begins where training ended.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..engine.types import RolloutResult

_M_TO_CM = 100.0
OSC_RMS_THRESHOLD_CMS = 10.0     # hold-RMS at/above this => the hold is oscillating (Δc trigger)


def _hold_slice(res: RolloutResult, hold_start_s: Optional[float]) -> slice:
    """Trajectory index slice for the hold window. None => trained horizon."""
    return slice(res.hold_start_index(hold_start_s), None)


def _signed_error_coords(res: RolloutResult) -> tuple[np.ndarray, np.ndarray]:
    """Signed (along, lateral) fingertip error in the reach frame, over time. Both [T+1, B], m.

    along   = signed error along the start->target axis (<0 short of target, >0 beyond it)
    lateral = signed perpendicular error (the tremor axis)
    Signed coordinates keep the oscillation's sign, so its true frequency is recoverable.
    """
    traj = np.asarray(res.trajectories, dtype=float)               # [T+1, B, 2]
    tgt = np.asarray(res.targets, dtype=float)                     # [B, 2]
    start = traj[0]                                                # [B, 2]
    axis = tgt - start
    u = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9, None)  # unit reach axis
    rel = traj - tgt[None]                                         # error rel. to target [T+1, B, 2]
    along = rel[..., 0] * u[None, :, 0] + rel[..., 1] * u[None, :, 1]
    lateral = u[None, :, 0] * rel[..., 1] - u[None, :, 1] * rel[..., 0]
    return along, lateral


def hold_speed_rms_cms(res: RolloutResult, *, hold_start_s: Optional[float] = None) -> np.ndarray:
    """[B] RMS fingertip speed over the hold window (cm/s) — the tremor AMPLITUDE / Δc trigger.

    A settled hold has speed ~0; a limit cycle sustains nonzero speed. RMS over the hold is
    a phase-agnostic amplitude measure (unlike a single terminal-speed sample)."""
    traj = np.asarray(res.trajectories, dtype=float)
    speed = np.linalg.norm(np.diff(traj, axis=0) / res.dt, axis=-1) * _M_TO_CM   # [T, B]
    hold = _hold_slice(res, hold_start_s)
    seg = speed[hold]
    if seg.shape[0] == 0:
        return np.full(res.n_trials, np.nan)
    return np.sqrt((seg ** 2).mean(axis=0))


def dominant_frequency_hz(res: RolloutResult, *, hold_start_s: Optional[float] = None
                          ) -> Dict[str, np.ndarray]:
    """Dominant hold oscillation frequency from the SIGNED lateral error (the tremor axis).

    Returns per-trial [B] arrays:
        freq_lateral_hz : peak-power frequency of the detrended lateral error over the hold
        freq_resolution_hz : Δf = 1 / T_hold (the spectral resolution; a scalar broadcast to [B])
    Detrend = subtract the hold-window mean (removes the steady-state bias, keeps the oscillation).
    Frequencies below one resolution bin are returned as NaN (unresolvable).
    """
    _, lateral = _signed_error_coords(res)                         # [T+1, B]
    hold = _hold_slice(res, hold_start_s)
    seg = lateral[hold] * _M_TO_CM                                 # [Th, B] cm
    Th, B = seg.shape
    if Th < 4:
        return {"freq_lateral_hz": np.full(B, np.nan),
                "freq_resolution_hz": np.full(B, np.nan)}
    seg = seg - seg.mean(axis=0, keepdims=True)                    # detrend (remove steady bias)
    window = np.hanning(Th)[:, None]
    power = np.abs(np.fft.rfft(seg * window, axis=0)) ** 2         # [F, B]
    freqs = np.fft.rfftfreq(Th, d=res.dt)                          # [F]
    power[0] = 0.0                                                 # kill DC
    peak = freqs[np.argmax(power, axis=0)]                         # [B]
    df = 1.0 / (Th * res.dt)                                       # resolution Δf
    peak = np.where(peak < df, np.nan, peak)                       # sub-resolution => unresolvable
    return {"freq_lateral_hz": peak, "freq_resolution_hz": np.full(B, df)}


def is_oscillating(res: RolloutResult, *, hold_start_s: Optional[float] = None) -> np.ndarray:
    """[B] boolean: hold-RMS at/above the Δc threshold (the hold has broken into oscillation)."""
    return hold_speed_rms_cms(res, hold_start_s=hold_start_s) >= OSC_RMS_THRESHOLD_CMS


def oscillation_metrics(res: RolloutResult, *, hold_start_s: Optional[float] = None
                        ) -> Dict[str, np.ndarray]:
    """The full R2 per-trial battery: amplitude (speed-based) + frequency (signed-based),
    ring-fenced as §4 requires."""
    freq = dominant_frequency_hz(res, hold_start_s=hold_start_s)
    return {
        "hold_speed_rms_cms": hold_speed_rms_cms(res, hold_start_s=hold_start_s),
        "freq_lateral_hz":    freq["freq_lateral_hz"],
        "freq_resolution_hz": freq["freq_resolution_hz"],
    }