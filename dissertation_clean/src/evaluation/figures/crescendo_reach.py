"""F10 — reach-phase intention-tremor crescendo. PURE matplotlib. Two panels:
(A) 'you can see it': lateral error vs normalised deceleration time, mean±SD across trials, one line
    per scenario — the swing widening on approach for the destabilising manipulations.
(B) quantified: peak-to-peak amplitude vs deceleration-phase bin, mean±SD across seeds, per scenario,
    with the growth ratio annotated.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .style import QUALITATIVE


def plot_crescendo_reach(scenarios: dict, *,
                         title="Reach-phase tremor: amplitude across deceleration") -> Figure:
    """scenarios: label -> {'time_norm':[nb], 'amp_mean':[nb], 'amp_sd':[nb], 'growth':(m,sd)}.
    One panel: amplitude-vs-deceleration-bin, mean±SD across seeds, per scenario."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (label, d) in enumerate(scenarios.items()):
        c = QUALITATIVE[i % len(QUALITATIVE)]
        x = d["time_norm"]
        ax.plot(x, d["amp_mean"], "o-", color=c, lw=2, ms=5,
                label=f"{label}  (growth {d['growth'][0]:.1f}±{d['growth'][1]:.1f}×)")
        ax.fill_between(x, d["amp_mean"] - d["amp_sd"], d["amp_mean"] + d["amp_sd"],
                        color=c, alpha=0.15)
    ax.set_xlabel("deceleration phase  (0 = peak velocity → 1 = target arrival)")
    ax.set_ylabel("lateral swing amplitude (cm, peak-to-peak)")
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8.5, title="manipulation")
    ax.grid(True, alpha=0.25)
    return fig