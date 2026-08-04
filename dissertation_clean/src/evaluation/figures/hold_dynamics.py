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
    res: RolloutResult,
    *,
    hold_start_s: Optional[float] = None,
    trial: int = 0,
    n_example_trials: int = 5,
    coord_limit_cm: Optional[float] = None,   # symmetric ± limit for the spatial/phase axes
    vel_limit_cms: Optional[float] = None,     # symmetric ± limit for the velocity axes
    title: str = "",
) -> Figure:
    """Five-panel hold diagnostic for ONE rollout (single trial for the phase planes/spatial plane;
    several trials for the time series). Pass ``coord_limit_cm`` / ``vel_limit_cms`` to fix frames
    across delays for a like-for-like comparison."""
    along, lateral, t = _hold_coords_cm(res, hold_start_s)
    B = res.n_trials
    b = trial
    rms = float(np.nanmean(hold_speed_rms_cms(res, hold_start_s=hold_start_s)))

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.32)
    ax_sp = fig.add_subplot(gs[0, 0])     # (c) spatial error-plane
    ax_pl = fig.add_subplot(gs[0, 1])     # (a) lateral phase plane
    ax_pa = fig.add_subplot(gs[0, 2])     # (a) along phase plane
    ax_ta = fig.add_subplot(gs[1, :])     # placeholder; split below into two
    ax_ta.remove()
    gs2 = gs[1, :].subgridspec(1, 2, wspace=0.22)
    ax_tal = fig.add_subplot(gs2[0, 0])   # (d) along vs t
    ax_tlat = fig.add_subplot(gs2[0, 1])  # (d) lateral vs t

    # ---- (c) spatial error-plane: lateral (x) vs along (y), target at origin ----
    sc = ax_sp.scatter(lateral[:, b], along[:, b], c=t, cmap=_TIME_CMAP, s=9, zorder=3)
    ax_sp.plot(lateral[:, b], along[:, b], color="0.7", lw=0.5, zorder=2)
    ax_sp.plot(0, 0, "o", mfc="none", mec="k", ms=12, mew=1.4, zorder=4, label="target")
    ax_sp.axhline(0, color="0.6", lw=0.7); ax_sp.axvline(0, color="0.6", lw=0.7)
    ax_sp.set_xlabel("lateral error (cm)"); ax_sp.set_ylabel("along-axis error (cm)")
    ax_sp.set_title("Spatial error (target at origin)\nsettle = point · limit cycle = loop")
    ax_sp.set_aspect("equal", adjustable="datalim")
    if coord_limit_cm:
        ax_sp.set_xlim(-coord_limit_cm, coord_limit_cm); ax_sp.set_ylim(-coord_limit_cm, coord_limit_cm)
    fig.colorbar(sc, ax=ax_sp, label="time (s)", fraction=0.046, pad=0.04)

    # ---- (a) phase planes: each axis vs its own velocity ----
    def _phase(ax, x_cm, name):
        v = np.gradient(x_cm[:, b], res.dt)              # cm/s
        p = ax.scatter(x_cm[:, b], v, c=t, cmap=_TIME_CMAP, s=8, zorder=3)
        ax.plot(x_cm[:, b], v, color="0.7", lw=0.5, zorder=2)
        ax.axhline(0, color="0.6", lw=0.7); ax.axvline(0, color="0.6", lw=0.7)
        ax.set_xlabel(f"{name} error (cm)"); ax.set_ylabel(f"{name} velocity (cm/s)")
        ax.set_title(f"{name.capitalize()} phase plane\n(closed loop = limit cycle)")
        if coord_limit_cm:
            ax.set_xlim(-coord_limit_cm, coord_limit_cm)
        if vel_limit_cms:
            ax.set_ylim(-vel_limit_cms, vel_limit_cms)
        return p
    _phase(ax_pl, lateral, "lateral")
    _phase(ax_pa, along, "along-axis")

    # ---- (d) time series: along vs t and lateral vs t, several trials ----
    step = max(1, B // n_example_trials)
    ex = list(range(0, B, step))[:n_example_trials]
    for ax, data, name, col in ((ax_tal, along, "along-axis", OKABE_ITO["blue"]),
                                (ax_tlat, lateral, "lateral", OKABE_ITO["vermillion"])):
        ax.axhline(0, color="0.6", lw=0.8)
        for bb in ex:
            ax.plot(t, data[:, bb], lw=1.0, alpha=0.85)
        ax.set_xlabel("time (s)"); ax.set_ylabel(f"{name} error (cm)")
        ax.set_title(f"{name.capitalize()} error over time ({len(ex)} trials)")
        if coord_limit_cm:
            ax.set_ylim(-coord_limit_cm, coord_limit_cm)

    sup = title or f"{res.architecture} · hold-phase speed RMS = {rms:.1f} cm/s"
    fig.suptitle(sup, fontsize=13)
    return fig


def plot_hold_trajectory_3d(
    res: RolloutResult,
    *,
    direction: int = 0,
    n_trials: int = 3,
    full_episode: bool = True,
    elev: float = 18.0,
    azim: float = -60.0,
    title: str = "",
) -> Figure:
    """APPENDIX (x, y, t) helix for ONE reach direction, coloured by TIME (so the spiral-up-then-
    settle vs persistent-helix reads off the colour). A tightening helix that becomes a vertical
    line = the hand stabilised; a helix that persists/widens up the time axis = sustained tremor.

    ``direction`` selects a target sector; ``n_trials`` trials from it are drawn. ``full_episode``
    shows the whole reach+hold (the settling is the point); set False for hold only."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    traj = np.asarray(res.trajectories, float)           # [T+1, B, 2] m
    ti = np.asarray(res.target_idx).astype(int)
    t = res.traj_time_s
    idx = np.where(ti == direction)[0][:n_trials]
    if idx.size == 0:
        raise ValueError(f"no trials for direction {direction}")

    lo = 0 if full_episode else res.hold_start_index(None)
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    for k, b in enumerate(idx):
        x, y = traj[lo:, b, 0] * 100.0, traj[lo:, b, 1] * 100.0
        c = QUALITATIVE[k % len(QUALITATIVE)]
        ax.plot(x, y, t[lo:], color=c, lw=1.2, alpha=0.9, label=f"trial {b}")
    ax.legend(fontsize=8, loc="upper left")
    # target as a vertical line across the whole time axis (a spatial point has no single 'time')
    tgt = np.asarray(res.targets, float)[idx[0]] * 100.0
    ax.plot([tgt[0], tgt[0]], [tgt[1], tgt[1]], [t[lo], t[-1]],
            color="red", lw=1.2, ls="--", alpha=0.7, label="target (x,y)")
    start = traj[lo, idx[0]] * 100.0
    ax.scatter(start[0], start[1], t[lo], marker="x", color="k", s=60)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_zlabel("time (s)", labelpad=12)
    fig.set_layout_engine("none")     # 3-D + tight_layout clips the z-label; disable it here
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title or f"{res.architecture} · direction {direction} · fingertip path (x, y, t)")
    return fig