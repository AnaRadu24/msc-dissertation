"""Feedback-ablation deviation time course (Layer 5, Figure Factory) — PURE matplotlib.

"When does removing feedback change the movement?" For each frozen channel, the
deviation of the frozen rollout's fingertip path from the *intact* path of the same
model (same seed, same targets), as a function of time. The reference is the intact
trajectory — not the target — so the curve isolates the causal effect of the
ablation (how much the movement *changed*), free of the intact reach's own residual
error.

This complements ``ablation.ablation_cascade_figure``: the bars give the endpoint
magnitude; this gives the *dynamics* — the onset latency and growth rate of the
collapse, which is a second, independent readout of "corrections are clocked by
feedback arrival". Its value is entirely in the time axis (the endpoint duplicates
the terminal-error bars by construction).

Pure: imports only numpy, matplotlib and ``.style``. All rollouts/analysis happen
upstream; deviations arrive here as ``(n_seeds, T)`` arrays already reduced over
trials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import OKABE_ITO, apply_style, save_figure

__all__ = ["DeviationSeries", "deviation_timecourse_figure"]

# Channel-freeze roles, coloured consistently with the loop-speed panels
# (proprioception warm/red, vision green) so the reader's learned mapping holds
# across the whole ablation section. "both" gets a distinct third colour.
_CONDITIONS = ("vision", "prop", "both")
_LABEL = {"vision": "vision freeze", "prop": "prop freeze", "both": "vision + prop freeze"}
_COL = {"vision": OKABE_ITO["sky_blue"], "prop": OKABE_ITO["orange"], "both": OKABE_ITO["purple"]}


@dataclass(frozen=True)
class DeviationSeries:
    """Per-seed deviation-from-intact (cm) time course for ONE architecture.

    ``t``           : (T,) time axis in seconds (shared across conditions/seeds).
    ``seeds``       : the seed ids (for the n in the title; curves are across-seed).
    ``per_seed[c]`` : (n_seeds, T) deviation in cm, already averaged over trials
                      within each seed. ``c`` is one of ``_CONDITIONS``.
    """
    t: np.ndarray
    seeds: Sequence[int]
    per_seed: Mapping[str, np.ndarray]

    def arr(self, cond: str) -> np.ndarray:
        return np.asarray(self.per_seed[cond], float)


def _plot_panel(ax, dev: DeviationSeries, title: str,
                delays_ms: Optional[Mapping[str, float]] = None) -> None:
    """One architecture: across-seed mean line + SD band per frozen channel."""
    for cond in _CONDITIONS:
        a = dev.arr(cond)                                   # (n_seeds, T)
        m = np.nanmean(a, axis=0)
        s = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(m)
        ax.plot(dev.t, m, color=_COL[cond], lw=2.2, label=_LABEL[cond], zorder=3)
        ax.fill_between(dev.t, m - s, m + s, color=_COL[cond], alpha=0.15, zorder=1)

    # optional faint guides at each channel's feedback delay (onset lags these,
    # because the arm must first move for a stale signal to diverge — do NOT read
    # onset == delay off these; they are orientation only)
    if delays_ms:
        for cond, d in delays_ms.items():
            ax.axvline(d / 1000.0, color=_COL.get(cond, "0.5"), ls=":", lw=1, alpha=0.6, zorder=0)

    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.margins(x=0)


def deviation_timecourse_figure(dual: DeviationSeries,
                                mono: Optional[DeviationSeries] = None, *,
                                delays_ms: Optional[Mapping[str, float]] = None,
                                paths=None, out_dir=None,
                                save: bool = True, show: bool = True):
    """Deviation-from-intact vs time. One panel if only ``dual`` given (matches the
    single-architecture draft); two shared-axis panels (dual | mono) if ``mono`` is
    supplied, so the architecture difference is read off per channel.

    ``delays_ms`` e.g. ``{"prop": 50, "vision": 100}`` draws faint orientation guides.
    """
    apply_style()

    if mono is None:
        fig, ax = plt.subplots(figsize=(9, 5.3))
        _plot_panel(ax, dual, f"dual (n={len(dual.seeds)})", delays_ms)
        ax.set_ylabel("deviation from intact reach (cm)")
        ax.legend(fontsize=9, loc="upper left")
        axes = [ax]
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
        _plot_panel(a1, dual, f"dual (n={len(dual.seeds)})", delays_ms)
        _plot_panel(a2, mono, f"mono (n={len(mono.seeds)})", delays_ms)
        a1.set_ylabel("deviation from intact reach (cm)")
        a1.legend(fontsize=9, loc="upper left")
        axes = [a1, a2]

    fig.suptitle("When does removing feedback change the movement?  (freeze from reset)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save:
        d = out_dir if out_dir is not None else paths.aggregated_dir("R1", create=True)
        save_figure(fig, d / "fig_R1_ablation_deviation", formats=("png",), close=False)
    if show:
        plt.show()
    return {"fig_R1_ablation_deviation": fig}