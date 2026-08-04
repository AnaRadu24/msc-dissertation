"""F5 — the money plot: delay-induced instability, architecture sets the threshold. PURE matplotlib.

The headline R2 figure. Two things on one axis, answering two questions:
  (1) per-SEED hold-RMS-vs-delay curves (thin) — the family of Hopf-like transitions and how sharply
      each breaks. NOT a mean-over-seeds curve: seeds differ in their CROSSING (Δc), not just
      amplitude, so averaging y at fixed x would blur the transition and produce meaningless bars.
  (2) Δc per architecture as a point + error bar (mean ± SD across seeds of each seed's Δc) on the
      delay axis — the threshold, which is what the margin claim actually rests on.

Reads DelaySweepResult objects (from sweep.delay_sweep_group); computes nothing itself beyond the
across-seed Δc summary. Colours from the architecture palette.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .style import OKABE_ITO, arch_color


def _arch_of(sweeps) -> str:
    """Infer architecture label from the model name for colouring ('dual'/'mono')."""
    return "dual" if sweeps[0].model.startswith("dual") else "mono"


def plot_money_plot(
    sweeps_by_arch: dict,
    *,
    threshold_cms: Optional[float] = None,
    log_y: bool = True,
    xlim: Optional[tuple] = None,
    ddof: int = 1,
    show_seed_labels: bool = True,
    title: str = "Delay-induced instability: architecture sets the critical delay",
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    thr = threshold_cms

    for label, sweeps in sweeps_by_arch.items():
        arch = _arch_of(sweeps)
        color = arch_color(arch)
        thr = thr or sweeps[0].threshold
        for i, s in enumerate(sweeps):
            y = np.maximum(s.y, 1e-3)
            ax.plot(s.x, y, "o-", color=color, lw=1.0, ms=3, alpha=0.55, zorder=2)
            if show_seed_labels:                       # faint seed number at the right end
                ends = sorted([(s.x[-1], max(s.y[-1], 1e-3), s.seed) for s in sweeps],
                          key=lambda e: e[1])
                # nudge each label to at least a fixed log-distance from the previous
                prev = 0
                for x_end, y_end, sd in ends:
                    y_lab = max(y_end, prev * 1)
                    ax.annotate(f"s{sd}", (x_end, y_lab), xytext=(4, 0), textcoords="offset points",
                                fontsize=3.5, color=color, alpha=0.85, va="center")
                    prev = y_lab
        # Δc summary -> into the legend label
        dcs = np.array([s.delta_c for s in sweeps], float)
        finite = dcs[np.isfinite(dcs)]
        m = float(np.mean(finite)) if finite.size else np.nan
        sd = float(np.std(finite, ddof=ddof)) if finite.size > 1 else np.nan
        n, ntot = int(finite.size), len(sweeps)
        dc_txt = (f"Δc = {m:.0f} ± {sd:.0f} ms" if np.isfinite(sd)
                  else f"Δc = {m:.0f} ms") + (f", n={n}" if n == ntot else f", n={n}/{ntot}")
        # a single proxy line for the legend, carrying the arch colour + Δc
        ax.plot([], [], "o-", color=color, ms=4, label=f"{label}  —  {dc_txt}")
        # a vertical band marking mean Δc ± sd on the delay axis
        if np.isfinite(m):
            ax.axvspan(m - (sd if np.isfinite(sd) else 0), m + (sd if np.isfinite(sd) else 0),
                       color=color, alpha=0.08, zorder=0)
            ax.axvline(m, color=color, ls="--", lw=1.2, alpha=0.7, zorder=1)

    ax.axhline(thr, ls=":", lw=1.2, color="0.4", zorder=1)
    ax.text(0.99, thr, f"onset {thr:g} cm/s ", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8, color="0.4")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("added feedback delay Δ (ms)")
    ax.set_ylabel("hold-phase speed RMS (cm/s)" + ("  [log]" if log_y else ""))
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    return fig