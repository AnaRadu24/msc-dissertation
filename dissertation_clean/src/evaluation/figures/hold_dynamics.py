"""Hold-phase dynamics figures (Layer 4) — PURE matplotlib. No torch, no rollout, no stored metrics.

Purpose: VALIDATE the hold-RMS order parameter by eye — is a high hold-RMS a genuine sustained
LIMIT CYCLE, a decaying transient, or a drift? Signed reach-frame coordinates (along = parallel to
the reach, >0 beyond target; lateral = perpendicular, the tremor axis) come from the ANALYSER
(oscillation._signed_error_coords) and are only DRAWN here.

plot_hold_dynamics — five diagnostic panels for ONE rollout:
    (c) spatial error-plane : lateral(x) vs along(y), target at origin, coloured by time
        -> settle = collapse to origin; limit cycle = closed loop; drift = march off-axis
    (a) two phase planes    : lateral vs d(lateral)/dt, and along vs d(along)/dt
        -> inward spiral = damped/stable; closed loop = limit cycle; outward = growing
    (d) two time series     : along vs t and lateral vs t, several example trials
        -> WHEN it settles, and whether trials/directions differ near threshold

plot_hold_trajectory_3d — appendix (x, y, t) helix for ONE direction, coloured by time.

All panels accept an optional shared coordinate limit so a stable and an unstable delay can be drawn
on IDENTICAL frames and compared directly (an 85x error-range difference otherwise makes the stable
case a dot and the unstable one fill the frame).
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..analyzers.oscillation import _signed_error_coords, hold_speed_rms_cms
from ..engine.types import RolloutResult
from .style import OKABE_ITO, QUALITATIVE

_M_TO_CM = 100.0
_TIME_CMAP = "viridis"


def _hold_coords_cm(res: RolloutResult, hold_start_s: Optional[float]):
    """(along_cm, lateral_cm, t) over the HOLD window. Both [Th, B]; t [Th]."""
    along, lateral = _signed_error_coords(res)           # [T+1, B] m
    h0 = res.hold_start_index(hold_start_s)
    return along[h0:] * _M_TO_CM, lateral[h0:] * _M_TO_CM, res.traj_time_s[h0:]


def plot_hold_dynamics(
    res, *, hold_start_s=None, start_s=0.0, trial=0,
    example_trials=None, n_example_trials=5,
    coord_limit_cm=None, vel_limit_cms=None, title="",
):
    from .style import QUALITATIVE
    along_full, lateral_full = _signed_error_coords(res)
    along_full = along_full * _M_TO_CM; lateral_full = lateral_full * _M_TO_CM
    lo = int(round(start_s / res.dt))
    along, lateral, t = along_full[lo:], lateral_full[lo:], res.traj_time_s[lo:]
    B = res.n_trials; b = trial
    i1 = int(round((1.0 - start_s) / res.dt))          # index of t=1s within the sliced window
    rms = float(np.nanmean(hold_speed_rms_cms(res, hold_start_s=hold_start_s)))

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.34)
    ax_sp = fig.add_subplot(gs[0,0]); ax_pl = fig.add_subplot(gs[0,1]); ax_pa = fig.add_subplot(gs[0,2])
    gs2 = gs[1,:].subgridspec(1, 2, wspace=0.24)
    ax_tal = fig.add_subplot(gs2[0,0]); ax_tlat = fig.add_subplot(gs2[0,1])

    def _startx(ax, x0, y0):                            # fine grey start X (no bulk)
        ax.plot(x0, y0, "x", color="0.4", ms=6, mew=0.8, zorder=6, label="start")

    # ---- (c) spatial error ----
    sc = ax_sp.scatter(lateral[:,b], along[:,b], c=t, cmap=_TIME_CMAP, s=9, zorder=3)
    ax_sp.plot(lateral[:,b], along[:,b], color="0.7", lw=0.5, zorder=2)
    # ax_sp.plot(0, 0, "x", color="red", ms=9, mew=2, zorder=6, label="target")
    if i1 < along.shape[0]:
        ax_sp.plot(lateral[i1,b], along[i1,b], "o", color="red", ms=6, mec="k", mew=0.5, zorder=7, label="t = 1 s")
    _startx(ax_sp, lateral[0,b], along[0,b])
    ax_sp.axhline(0,color="0.6",lw=0.7); ax_sp.axvline(0,color="0.6",lw=0.7)
    ax_sp.set_xlabel("lateral error (cm)"); ax_sp.set_ylabel("along-axis error (cm)")
    ax_sp.set_title("Spatial error (target at origin)"); ax_sp.set_aspect("equal", adjustable="datalim")
    ax_sp.legend(fontsize=7, loc="best")
    fig.colorbar(sc, ax=ax_sp, label="time (s)", fraction=0.046, pad=0.04)

    # ---- (a) phase planes, each with its own colourbar + t=1s marker + start X ----
    def _phase(ax, x_cm, name):
        v = np.gradient(x_cm[:,b], res.dt)
        p = ax.scatter(x_cm[:,b], v, c=t, cmap=_TIME_CMAP, s=8, zorder=3)
        ax.plot(x_cm[:,b], v, color="0.7", lw=0.5, zorder=2)
        if i1 < x_cm.shape[0]:
            ax.plot(x_cm[i1,b], v[i1], "o", color="red", ms=6, mec="k", mew=0.5, zorder=7, label="t = 1 s")
        _startx(ax, x_cm[0,b], v[0])
        ax.axhline(0,color="0.6",lw=0.7); ax.axvline(0,color="0.6",lw=0.7)
        ax.set_xlabel(f"{name} error (cm)"); ax.set_ylabel(f"{name} velocity (cm/s)")
        ax.set_title(f"{name.capitalize()} phase plane\n(closed loop = limit cycle)")
        ax.legend(fontsize=7, loc="best")
        fig.colorbar(p, ax=ax, label="time (s)", fraction=0.046, pad=0.04)
        if coord_limit_cm: ax.set_xlim(-coord_limit_cm, coord_limit_cm)
        if vel_limit_cms: ax.set_ylim(-vel_limit_cms, vel_limit_cms)
    _phase(ax_pl, lateral, "lateral")
    _phase(ax_pa, along, "along-axis")

    # ---- (d) time series: same trials, legend shows trial + its error at t=1s ----
    if example_trials is None:
        step = max(1, B // n_example_trials)
        example_trials = list(range(0, B, step))[:n_example_trials]
    for ax, data, name in ((ax_tal, along, "along-axis"), (ax_tlat, lateral, "lateral")):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(1.0, color="red", ls="--", lw=1.0, alpha=0.7)      # end of trained 1 s
        for k, bb in enumerate(example_trials):
            e1 = data[i1, bb] if i1 < data.shape[0] else np.nan       # this axis's error at t=1s
            ax.plot(t, data[:,bb], lw=1.2, alpha=0.9, color=QUALITATIVE[k % len(QUALITATIVE)],
                    label=f"trial {bb}: {e1:+.2f} cm @1s")
        ax.set_xlabel("time (s)"); ax.set_ylabel(f"{name} error (cm)")
        ax.set_title(f"{name.capitalize()} error over time ({len(example_trials)} trials)")
        ax.legend(fontsize=7, loc="best")
        if coord_limit_cm: ax.set_ylim(-coord_limit_cm, coord_limit_cm)

    fig.suptitle(title or f"{res.architecture} · hold-phase speed RMS = {rms:.1f} cm/s", fontsize=13)
    fig.set_layout_engine("none")                        # same autolayout defeat as the composite
    fig.subplots_adjust(left=0.06, right=0.97, top=0.91, bottom=0.08)
    return fig

def plot_hold_trajectory_3d(res, *, direction=0, n_trials=3, full_episode=True,
                            elev=18.0, azim=-60.0, title=""):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from .style import QUALITATIVE
    traj = np.asarray(res.trajectories, float); tgt = np.asarray(res.targets, float)
    ti = np.asarray(res.target_idx).astype(int); t = res.traj_time_s
    idx = np.where(ti == direction)[0][:n_trials]
    if idx.size == 0: raise ValueError(f"no trials for direction {direction}")
    lo = 0 if full_episode else res.hold_start_index(None)
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    for k, b in enumerate(idx):
        c = QUALITATIVE[k % len(QUALITATIVE)]
        x, y = traj[lo:, b, 0]*100, traj[lo:, b, 1]*100
        term = np.linalg.norm(traj[-1, b] - tgt[b]) * 100          # terminal error, cm
        ax.plot(x, y, t[lo:], color=c, lw=1.2, alpha=0.9, label=f"trial {b}  (term {term:.1f} cm)")
        tg = tgt[b]*100
        ax.plot([tg[0], tg[0]], [tg[1], tg[1]], [t[lo], t[-1]], color=c, ls=":", lw=1.5, alpha=0.8, label=f"target trial {b}")
    start = traj[lo, idx[0]]*100
    ax.scatter(start[0], start[1], t[lo], marker="x", color="k", s=60, label="start")
    ax.set_xlabel("x (cm)", labelpad=8); ax.set_ylabel("y (cm)", labelpad=8); ax.set_zlabel("time (s)", labelpad=12)
    ax.set_zlim(t[lo], t[-1])                                       # kill the phantom negative padding
    ax.view_init(elev=elev, azim=azim); fig.set_layout_engine("none")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title or f"{res.architecture} · direction {direction} · fingertip path (x, y, t)")
    return fig

def plot_reach_2d_by_time(res, *, direction=0, n_trials=3, from_s=0.0, title=""):
    """2D fingertip path in workspace (x,y), coloured by time, for n_trials of ONE direction (same
    target, guaranteed). Full episode from from_s. The target is the shared target of that direction."""
    import numpy as np, matplotlib.pyplot as plt
    t1_label = "t = 1s"
    target_label = "target"
    traj = np.asarray(res.trajectories); tgt = np.asarray(res.targets); dt = res.dt
    ti = np.asarray(res.target_idx).astype(int)
    idx = np.where(ti == direction)[0][:n_trials]
    if idx.size == 0: raise ValueError(f"no trials for direction {direction}")
    idx_t1 = int(round(1.0 / dt))
    lo = int(round(from_s/dt))
    t = np.arange(traj.shape[0])*dt
    fig, ax = plt.subplots(figsize=(6.5,6))
    for b in idx:
        sc = ax.scatter(traj[lo:,b,0]*100, traj[lo:,b,1]*100, c=t[lo:], cmap="viridis", s=8, zorder=3)
        ax.plot(traj[lo:,b,0]*100, traj[lo:,b,1]*100, color="0.8", lw=0.4, zorder=2)
        # Add the red dot if the index falls within the active trajectory data
        if lo <= idx_t1 < traj.shape[0]:
            ax.plot(traj[idx_t1, b, 0]*100, traj[idx_t1, b, 1]*100, "o", 
                    color="red", ms=5, mec="k", mew=0.5, label=t1_label, zorder=6)
            t1_label = None # Prevents duplicate legend items for subsequent trials
        tg = tgt[b]*100
        ax.plot(tg[0], tg[1], "x", mfc="none", mec="red", ms=7, mew=2, label=target_label, zorder=4)
        target_label = None # Prevents duplicate legend items for subsequent trials
           
    ax.plot(traj[lo,idx[0],0]*100, traj[lo,idx[0],1]*100, "kx", ms=10, label="start", zorder=4)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, label="time (s)", fraction=0.046, pad=0.04)
    cb.ax.axhline(1, color="red", linewidth=2, zorder=5)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig