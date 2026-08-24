"""Command-divergence figures (R1, Layer 5) — PURE matplotlib.
 
Establishes that corrections are REACTIVE, clocked by feedback arrival: the muscle
commands for a perturbed and an unperturbed trial (identical target/seed, differing
only in the curl) are bit-identical until the shortest available feedback delay,
then diverge abruptly. Two figures:
 
  divergence_scatter_figure  — measured divergence latency vs predicted = min(available
                               delay), across the three delay assignments and the four
                               ablations; points on the identity line. MAIN TEXT.
  divergence_trace_figure    — one config's muscle-excitation traces (perturbed solid /
                               unperturbed dashed) with the log |Δ| jump at feedback
                               arrival, plus onset / predicted / measured markers. APPENDIX.
 
Pure: numpy + matplotlib + .style only. The measured latency and the |Δ| series are
computed by analyzers.divergence; this module only draws what it is handed.
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence
 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
 
from .style import OKABE_ITO, apply_style, save_figure
 
__all__ = ["DivergencePoint", "DivergenceTrace",
           "divergence_scatter_figure", "divergence_trace_figure"]
 
# Config colours match the loop-speed / causal panels; ablation encoded by marker
# so the reader reads "which model" off colour and "which freeze" off shape.
_CONFIG_C = {"baseline": OKABE_ITO["bluish_green"], "matched": OKABE_ITO["yellow"], "swapped": OKABE_ITO["purple"]}
 
_ABLATION_M = {"intact": "o", "vision": "s", "prop": "^", "both": "X"}
_ABLATION_LABEL = {"intact": "intact", "vision": "vision frozen",
                   "prop": "prop frozen", "both": "both frozen"}
 
 
@dataclass(frozen=True)
class DivergencePoint:
    """One (config, ablation) measurement for the identity-line scatter."""
    config: str            # "baseline" | "matched" | "swapped"
    ablation: str          # "intact" | "vision" | "prop" | "both"
    predicted_ms: float    # min available feedback delay (the prediction)
    measured_ms: float     # measured command-divergence latency
 
 
@dataclass(frozen=True)
class DivergenceTrace:
    """One config's perturbed-vs-unperturbed command traces for the appendix panel."""
    t_ms: np.ndarray                    # (T,) time from movement start, ms
    excitation_perturbed: np.ndarray    # (T, k) a few illustrative muscles
    excitation_unperturbed: np.ndarray  # (T, k)
    muscle_labels: Sequence[str]        # length k
    max_abs_delta: np.ndarray           # (T,) max over ALL muscles of |pert − unpert|
    onset_ms: float                     # movement onset
    predicted_ms: float                 # onset + min available delay
    measured_ms: float                  # measured divergence
    muscle_colours: Sequence[str] = field(default_factory=lambda: ["#C44E52", "#4C72B0"])
 
 
# ===========================================================================
# Main-text: the identity-line scatter.
# ===========================================================================
def divergence_scatter_figure(points: Sequence[DivergencePoint], *,
                              paths=None, out_dir=None,
                              save: bool = True, show: bool = True):
    """Measured vs predicted divergence latency. Points should sit on the identity
    line (measured = min available delay); when the fast channel is frozen the point
    jumps to the slower channel's latency — causal evidence the fast loop clocks it."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
 
    lo, hi = 0.0, max(max(p.predicted_ms, p.measured_ms) for p in points) * 1.15
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=1.2, zorder=1,
            label="identity (perfect prediction)")
 
    for p in points:
        ax.scatter(p.predicted_ms, p.measured_ms,
                   marker=_ABLATION_M.get(p.ablation, "o"),
                   color=_CONFIG_C.get(p.config, "0.3"),
                   s=90, edgecolor="k", linewidth=0.5, zorder=3)
 
    # two legends: colour = config, marker = ablation
    cfg_handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=c, markeredgecolor="k", markeredgewidth=0.5,
                          label=cfg) for cfg, c in _CONFIG_C.items()]
    abl_handles = [Line2D([0], [0], marker=m, linestyle="none", markersize=9,
                          markerfacecolor="0.6", markeredgecolor="k", markeredgewidth=0.5,
                          label=_ABLATION_LABEL[a]) for a, m in _ABLATION_M.items()]
    leg1 = ax.legend(handles=cfg_handles, title="delay assignment",
                     fontsize=8, title_fontsize=8, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=abl_handles, title="ablation", fontsize=8, title_fontsize=8,
              loc="lower right")
 
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("predicted latency = min(available delays) (ms)")
    ax.set_ylabel("measured divergence latency (ms)")
    ax.set_title("Response is clocked by the arrival of sensory information")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout()
 
    if save:
        d = out_dir if out_dir is not None else paths.aggregated_dir("R1", create=True)
        save_figure(fig, d / "fig_R1_divergence_scatter", formats=("png",), close=False)
    if show:
        plt.show()
    return {"fig_R1_divergence_scatter": fig}
 
 
# ===========================================================================
# Appendix: the perturbed-vs-unperturbed command trace.
# ===========================================================================
def divergence_trace_figure(trace: DivergenceTrace, *,
                            paths=None, out_dir=None,
                            save: bool = True, show: bool = True):
    """Two stacked panels for ONE config: raw muscle excitations (perturbed solid,
    unperturbed dashed) and, below, the max |Δ| on a log axis — flat at machine
    precision until feedback arrives, then jumping orders of magnitude."""
    apply_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))
 
    muscle_colors = [
        OKABE_ITO["blue"],         # SF
        OKABE_ITO["sky_blue"],     # SE
        OKABE_ITO["bluish_green"], # BF
        OKABE_ITO["yellow"],       # BE
        OKABE_ITO["purple"],       # EF
        OKABE_ITO["orange"]        # EE
    ]
    
    k = trace.excitation_perturbed.shape[1]
    for j in range(k):
        c = muscle_colors[j % len(muscle_colors)]
        lbl = trace.muscle_labels[j]
        ax_top.plot(trace.t_ms, trace.excitation_perturbed[:, j], color=c, lw=1.3,
                    label=f"{lbl} — perturbed")
        ax_top.plot(trace.t_ms, trace.excitation_unperturbed[:, j], color=c, lw=1.1,
                    ls="--", alpha=0.9, label=f"{lbl} — unperturbed")
 
    def _marks(ax):
        ax.axvline(trace.onset_ms, color="0.4", lw=1.0, label="movement onset")
        ax.axvline(trace.predicted_ms, color="#4C72B0", ls="--", lw=1.2,
                   label="onset + min delay (predicted)")
        ax.axvline(trace.measured_ms, color="k", ls=":", lw=1.2, label="measured divergence")
 
    _marks(ax_top)
    ax_top.set_ylabel("muscle excitation\n(network output)")
    ax_top.set_title("Perturbed (solid) vs unperturbed (dashed) — commands split at feedback arrival")
    ax_top.legend(fontsize=7, ncol=2, loc="upper right")
    ax_top.grid(True, alpha=0.3); ax_top.set_axisbelow(True)
 
    dd = np.asarray(trace.max_abs_delta, float)
    dd = np.where(dd <= 0, np.nan, dd)          # log axis: drop exact zeros
    ax_bot.semilogy(trace.t_ms, dd, color="#C44E52", lw=1.3)
    _marks(ax_bot)
    ax_bot.set_ylabel(r"max $|\Delta$ excitation$|$")
    ax_bot.set_xlabel("time from movement start (ms)")
    ax_bot.grid(True, which="both", alpha=0.3); ax_bot.set_axisbelow(True)
    ax_bot.margins(x=0)
 
    if save:
        d = out_dir if out_dir is not None else paths.aggregated_dir("R1", create=True)
        save_figure(fig, d / "fig_R1_divergence_trace", formats=("png",), close=False)
    if show:
        plt.show()
    return {"fig_R1_divergence_trace": fig}
 
# ===========================================================================
# Appendix: six-muscle perturbed-vs-unperturbed, shown as SIGNED DIFFERENCES.
# Two overlaid near-identical traces read as one line; the signed difference
# u_pert - u_unpert is flat-at-zero then departs, so "identical then diverges"
# becomes a visible event per muscle. Landmark = t_stim + Δ_min (see note), not
# movement onset, because the velocity-dependent curl is ~0 at onset.
# ===========================================================================
def divergence_muscles_figure(trace: DivergenceTrace, *,
                              window_ms: Optional[float] = 250.0,
                              stim_ms: Optional[float] = None,
                              paths=None, out_dir=None,
                              save: bool = True, show: bool = True):
    """Per-muscle command divergence for ONE config (appendix companion to the
    scatter). One small panel per muscle showing the SIGNED command difference
    u_pert - u_unpert (cm-free network units): flat at zero while the commands
    coincide, departing once the perturbation is detected through feedback.
 
    Landmarks drawn on every panel:
        stim_ms      : t_stim, when the curl-induced deviation first matters
                       (defaults to trace.onset_ms if not supplied; pass the real
                       perturbation-crosses-threshold time when available).
        predicted_ms : t_stim + Δ_min, the earliest a reactive controller could respond.
        measured_ms  : the measured divergence.
    window_ms crops the x-axis so the pre/at-divergence interval is legible (the
    hold oscillations that dominate a full episode are irrelevant here).
    """
    apply_style()
    t = np.asarray(trace.t_ms, float)
    P = np.asarray(trace.excitation_perturbed, float)
    U = np.asarray(trace.excitation_unperturbed, float)
    k = P.shape[1]
    diff = P - U                                     # (T, k) signed difference
 
    m = t <= window_ms if window_ms is not None else np.ones_like(t, bool)
    t_stim = trace.onset_ms if stim_ms is None else stim_ms
 
    muscle_colors = [OKABE_ITO["blue"], OKABE_ITO["sky_blue"], OKABE_ITO["bluish_green"],
                     OKABE_ITO["yellow"], OKABE_ITO["purple"], OKABE_ITO["orange"]]
 
    ncol = 3
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.5 * nrow),
                             sharex=True, squeeze=False)
    ymax = np.nanmax(np.abs(diff[m])) * 1.15 or 1e-3   # shared symmetric y-limit
 
    for j in range(k):
        ax = axes[j // ncol][j % ncol]
        c = muscle_colors[j % len(muscle_colors)]
        # shaded "commands identical" region: t_stim -> measured divergence
        ax.axvspan(t_stim, trace.measured_ms, color="0.85", alpha=0.45, zorder=0)
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.plot(t[m], diff[m, j], color=c, lw=1.6, zorder=3)
        ax.axvline(t_stim, color="0.35", lw=1.2, zorder=2)
        ax.axvline(trace.predicted_ms, color="#4C72B0", ls="--", lw=1.4, zorder=2)
        ax.axvline(trace.measured_ms, color="#C44E52", ls=":", lw=1.6, zorder=2)
        ax.set_ylim(-ymax, ymax)
        ax.set_title(trace.muscle_labels[j], fontsize=9)
        ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
        if j % ncol == 0:
            ax.set_ylabel(r"$u_{\mathrm{pert}}-u_{\mathrm{unpert}}$")
        if j // ncol == nrow - 1:
            ax.set_xlabel("time from movement onset (ms)")
 
    for j in range(k, nrow * ncol):                  # hide unused axes
        axes[j // ncol][j % ncol].set_visible(False)
 
    handles = [
        Line2D([0], [0], color="0.35", lw=1.2, label=r"$t_{\mathrm{stim}}$ (perturbation matters)"),
        Line2D([0], [0], color="#4C72B0", ls="--", lw=1.4,
               label=r"$t_{\mathrm{stim}}+\Delta_{\min}$ (earliest reactive response)"),
        Line2D([0], [0], color="#C44E52", ls=":", lw=1.6, label="measured divergence"),
        Line2D([0], [0], color="0.85", lw=8, alpha=0.6, label="commands identical"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-muscle command divergence: signed difference (perturbed − unperturbed)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
 
    if save:
        d = out_dir if out_dir is not None else paths.aggregated_dir("R1", create=True)
        save_figure(fig, d / "fig_R1_divergence_muscles", formats=("png",), close=False)
    if show:
        plt.show()
    return {"fig_R1_divergence_muscles": fig}