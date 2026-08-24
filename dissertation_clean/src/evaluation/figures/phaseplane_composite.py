"""R2 phase-plane composite — PURE matplotlib. For ONE trial of ONE model, four views over the full
episode: 2D workspace trajectory, spatial error (reach frame), lateral phase plane, along-axis phase
plane. All coloured by time, all the SAME trial (so target is unambiguous and panels are consistent).
Used to show the stable->limit-cycle transition across a few added delays."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

_M = 100.0
_CMAP = "viridis"


def _reach_frame(traj, tgt):
    """Signed (lateral, along) error in the reach frame, cm. traj [T+1,2], tgt [2]."""
    start = traj[0]; axis = tgt - start
    u = axis / max(np.linalg.norm(axis), 1e-9)
    rel = traj - tgt
    lateral = (u[0]*rel[:, 1] - u[1]*rel[:, 0]) * _M
    along = (u[0]*rel[:, 0] + u[1]*rel[:, 1]) * _M
    return lateral, along


def plot_phaseplane_composite(res, *, trial=0, title="") -> Figure:
    traj = np.asarray(res.trajectories)[:, trial]; tgt = np.asarray(res.targets)[trial]
    dt = res.dt; t = np.arange(traj.shape[0]) * dt
    lateral, along = _reach_frame(traj, tgt)
    dlat = np.gradient(lateral, dt); dalo = np.gradient(along, dt)
    i1 = int(round(1.0 / dt))                       # end of the trained 1 s episode

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    def _sc(ax, x, y):
        ax.plot(x, y, color="0.85", lw=0.3, zorder=1)
        p = ax.scatter(x, y, c=t, cmap=_CMAP, s=5, zorder=2)
        if i1 < len(x):                             # red marker = end of trained 1 s trial
            ax.plot(x[i1], y[i1], "o", color="red", ms=5, mec="k", mew=0.5, zorder=5, label="t = 1 s")
        return p
    # (1) workspace — both x and y as signed distances to the target
    ax = axes[0,0]
    # Calculate relative coordinates (signed distances) to the target
    x_signed_dist_cm = (traj[:, 0] - tgt[0]) * _M
    y_signed_dist_cm = (traj[:, 1] - tgt[1]) * _M
    _sc(ax, x_signed_dist_cm, y_signed_dist_cm)
    ax.plot(0, 0, "x", color="red", ms=8, zorder=6, label="target")
    ax.plot(x_signed_dist_cm[0], y_signed_dist_cm[0], "kx", ms=8, label="start")
    ax.axhline(0,color="0.6",lw=0.6); ax.axvline(0,color="0.6",lw=0.6)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_title("Signed distance to target"); 
    ax.legend(fontsize=8)
    
    # (2) spatial error — target at origin (red X at origin)
    ax = axes[0,1]; _sc(ax, lateral, along)
    ax.plot(0, 0, "x", color="red", ms=8, zorder=6, label="target")
    ax.plot(lateral[0], along[0], "kx", ms=8, label="start")
    ax.axhline(0,color="0.6",lw=0.6); ax.axvline(0,color="0.6",lw=0.6)
    ax.set_xlabel("lateral error (cm)"); ax.set_ylabel("along-axis error (cm)")
    ax.set_title("Spatial error (target at origin)")
    ax.legend(fontsize=8)
    # (3) lateral phase plane
    ax = axes[1,0]; _sc(ax, lateral, dlat)
    ax.axhline(0,color="0.6",lw=0.6); ax.axvline(0,color="0.6",lw=0.6)
    ax.plot(0, 0, "x", color="red", ms=8, zorder=6, label="target")
    ax.plot(lateral[0], dlat[0], "x", color="0.4", ms=6, mew=0.9, zorder=6, label="start")
    ax.set_xlabel("lateral error (cm)"); ax.set_ylabel("lateral velocity (cm/s)")
    ax.set_title("Lateral phase plane (closed loop = limit cycle)")
    ax.legend(fontsize=8)
    # (4) along-axis phase plane
    ax = axes[1,1]; p = _sc(ax, along, dalo)
    ax.plot(0, 0, "x", color="red", ms=8, zorder=6, label="target")
    ax.plot(along[0], dalo[0], "x", color="0.4", ms=6, mew=0.9, zorder=6, label="start")
    ax.axhline(0,color="0.6",lw=0.6); ax.axvline(0,color="0.6",lw=0.6)
    ax.set_xlabel("along-axis error (cm)"); ax.set_ylabel("along-axis velocity (cm/s)")
    ax.set_title("Along-axis phase plane")
    ax.legend(fontsize=8)

    fig.set_layout_engine("none")          # defeat apply_style's figure.autolayout=True, which otherwise repositions panels over the colourbar
    fig.subplots_adjust(left=0.07, right=0.86, top=0.93, bottom=0.07, wspace=0.28, hspace=0.30)
    cax = fig.add_axes([0.89, 0.15, 0.015, 0.7])
    fig.colorbar(p, cax=cax, label="time (s)")
    cax.axhline(1, color="red", linewidth=2, zorder=5)
    if title: fig.suptitle(title, fontsize=13)
    return fig