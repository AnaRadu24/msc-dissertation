"""Single-seed crescendo diagnostic — PURE matplotlib. Binned lateral amplitude + per-bin frequency
across the FULL 1 s window, with peak-velocity (decel onset) and phase-end markers. Shows the
amplitude growing on approach while frequency stays ~constant — the intention-tremor signature."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..analyzers.crescendo import _lateral_cm
from .style import OKABE_ITO


def plot_crescendo_diagnostic(res, deceleration_result, *, trial=0, n_bins_full=14,
                              window_ms=150.0, title="") -> Figure:
    """Bin the FULL 1 s into n_bins_full equal-time bins; per bin plot peak-to-peak lateral amplitude
    (bars) + dominant frequency (line). Vertical lines mark peak-velocity and phase-end (from
    deceleration_result). Amplitude should rise toward the target while frequency stays flat."""
    traj = np.asarray(res.trajectories); tgt = np.asarray(res.targets); dt = res.dt
    b = trial
    lateral = _lateral_cm(traj, tgt)[:, b]                        # [T+1] cm
    T = len(lateral); t = np.arange(T) * dt
    i_peak = int(deceleration_result["i_peak"][b]); i_end = int(deceleration_result["i_hold"][b])

    edges = np.linspace(0, T - 1, n_bins_full + 1).astype(int)
    centers_t, amp, freq = [], [], []
    for k in range(n_bins_full):
        lo, hi = edges[k], edges[k + 1]
        seg = lateral[lo:hi]
        if seg.size >= 3:
            centers_t.append(t[(lo + hi) // 2])
            amp.append(seg.max() - seg.min())
            s = seg - seg.mean()
            if s.std() > 0.02:
                f = np.fft.rfftfreq(len(s), dt); P = np.abs(np.fft.rfft(s))**2
                P[0] = 0; freq.append(f[np.argmax(P)])
            else:
                freq.append(np.nan)
    centers_t, amp, freq = np.array(centers_t), np.array(amp), np.array(freq)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(centers_t, amp, width=(t[-1]/n_bins_full)*0.85, color=OKABE_ITO["vermillion"],
           alpha=0.55, label="lateral swing amplitude (peak-to-peak)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("lateral amplitude (cm)", color=OKABE_ITO["vermillion"])
    ax.axvline(t[i_peak], color="0.3", ls="--", lw=1.2)
    ax.text(t[i_peak], ax.get_ylim()[1]*0.95, " peak vel\n (decel onset)", fontsize=8, va="top")
    ax.axvline(t[i_end], color="green", ls="--", lw=1.2)
    ax.text(t[i_end], ax.get_ylim()[1]*0.95, " phase end", fontsize=8, va="top", color="green")

    ax2 = ax.twinx()
    ax2.plot(centers_t, freq, "o-", color="purple", ms=4, lw=1.2, label="dominant frequency")
    ax2.set_ylabel("dominant frequency (Hz)", color="purple"); ax2.set_ylim(0, 5)
    ax2.tick_params(axis='y', labelcolor="purple")
    ax.set_title(title or f"{res.architecture} · trial {b} — amplitude grows, frequency ~constant")
    fig.tight_layout()
    return fig

def plot_reach_trajectory(res, *, trial=0, title="") -> Figure:
    """Spatial fingertip path in the reach frame (lateral x, along y), target at origin, coloured by
    time, over the FULL rollout. The tremor is visible as a widening/orbiting path near the target."""
    traj = np.asarray(res.trajectories); tgt = np.asarray(res.targets); dt = res.dt
    b = trial
    start = traj[0]; axis = tgt - start
    u = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9, None)
    rel = traj - tgt[None]
    lateral = (u[None, :, 0]*rel[..., 1] - u[None, :, 1]*rel[..., 0])[:, b] * 100.0
    along = (u[None, :, 0]*rel[..., 0] + u[None, :, 1]*rel[..., 1])[:, b] * 100.0
    t = np.arange(len(lateral)) * dt

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(lateral, along, c=t, cmap="viridis", s=10, zorder=3)
    ax.plot(lateral, along, color="0.7", lw=0.5, zorder=2)
    ax.plot(0, 0, "o", mfc="none", mec="k", ms=12, mew=1.4, label="target")
    ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("lateral error (cm)"); ax.set_ylabel("along-axis error (cm)")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=ax, label="time (s)", fraction=0.046, pad=0.04)
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    return fig