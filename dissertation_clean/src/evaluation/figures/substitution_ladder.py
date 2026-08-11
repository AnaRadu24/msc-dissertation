"""Substitution ladder figure (R3 mechanism: phase, not gain).

Rolls dual seeds out with the Task Net's estimate slot filled from a ladder of sources, sweeps an
injected delay Δ, and measures terminal error at the trained horizon. Two panels:

  F6a  ladder  — terminal error vs Δ for each source (the phase-lead story)
  F6b  bars    — a hand-picked set of (source, Δ) contrasts
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..analyzers.reach import terminal_error_cm
from ..engine import rollout
from ..engine.types import IDENTITY, Intervention
from ..manipulators.delay import delay
from ..manipulators.substitution import substitute
from ..settings import DEFAULT
from .style import QUALITATIVE, apply_style, save_figure

# --- configuration -------------------------------------------------------------------------------
GRID_MS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 150, 170, 200)
_PROF = DEFAULT.profile("competence")
_ROLLOUT_KW = {
    "task": "reach", 
    "duration_s": _PROF.duration_s, 
    "batch_size": _PROF.batch_size, 
    "seed": DEFAULT.rollout_seed
}

_DELAY_MODE = "additive_both"   
_BAR_DELTA_C = 50.0             

# ladder sources: key -> (legend label, QUALITATIVE colour index, swept?)
_LADDER_SOURCES = [
    ("instant_feedback",         "Instant feedback y(t) (perfect estimator)",          0, False),
    ("delayed_feedback",         "Predictor x̂(y(t−Δ_prop-Δ), u(t)) (trained model)",                        1, True),
    ("relayed_delayed_feedback", "Delayed feedback y(t−Δ_prop−Δ) (naive relay)",              2, True),
    ("delayed_estimate",         "Delayed estimate x̂(y(t−Δ_prop−Δ), u(t−Δ)) (stale prediction)",         3, True),
    ("zero",                     "Zero estimate (severed, OOD floor)",                 4, False),
    ("constant_predictor_output","Constant x̂ (in-distribution floor)",                 5, False),
]

# barplot contrasts: (label, source key, Δ ms). Labels reference _BAR_DELTA_C=50 literally.
_BARS = [
    ("Instant\nfeedback",       "instant_feedback",           0.0),
    ("Intact model\n(Δ=0)",     "delayed_feedback",           0.0),
    ("Predictor\n(Δ=50)",       "delayed_feedback",           _BAR_DELTA_C),
    ("Naive relay\n(Δ=0)",      "relayed_delayed_feedback",   0.0),
    ("Naive relay\n(Δ=50)",     "relayed_delayed_feedback",   _BAR_DELTA_C),
    ("Stale x̂\n(Δ=50)",         "delayed_estimate",           _BAR_DELTA_C),
    ("Zero\n(severed, OOD)",    "zero",                        0.0),
    ("Constant x̂\n(in-dist)",   "constant_predictor_output",   0.0),
]

# sources whose value does not depend on Δ (computed once, cached at Δ=0)
_DELTA_INDEPENDENT = {"instant_feedback", "zero", "constant_predictor_output"}

_COLOUR_IDX = {"instant_feedback": 0, "delayed_feedback": 1, "relayed_delayed_feedback": 2,
               "delayed_estimate": 3, "zero": 4, "constant_predictor_output": 5}


def _colour(source):
    return QUALITATIVE[_COLOUR_IDX[source] % len(QUALITATIVE)]


# --- intervention construction -------------------------------------------------------------------
def _combine(env_iv: Intervention, sub_iv: Intervention) -> Intervention:
    return Intervention(
        label=f"{env_iv.label}+{sub_iv.label}",
        description=f"{env_iv.description}; {sub_iv.description}",
        config_overrides={**env_iv.config_overrides, **sub_iv.config_overrides},
        estimate_source=sub_iv.estimate_source,
    )


def _iv(source: str, delta_ms: float, rec) -> Intervention:
    bp, bv = rec.config.proprioception_delay, rec.config.vision_delay
    env_delay = (IDENTITY if delta_ms == 0 else
                 delay(delta_ms, mode=_DELAY_MODE, base_prop_ms=bp, base_vision_ms=bv))

    if source == "delayed_feedback":                       
        return env_delay                                   
    if source == "relayed_delayed_feedback":               
        return _combine(env_delay, substitute("relayed_feedback"))
    if source == "delayed_estimate":                       
        return _combine(IDENTITY, substitute("delayed_estimate", estimate_delay_ms=delta_ms))
    if source in _DELTA_INDEPENDENT:                        
        return _combine(IDENTITY, substitute(source))
    raise ValueError(f"unknown ladder source {source!r}")


# --- evaluation (cached; one rollout per (source, effective Δ, seed)) ----------------------------
class _Evaluator:
    def __init__(self, group, paths):
        self.group, self.paths, self._cache = group, paths, {}

    def terminal(self, source: str, delta_ms: float) -> np.ndarray:
        eff = 0.0 if source in _DELTA_INDEPENDENT else float(delta_ms)
        key = (source, eff)
        if key not in self._cache:
            vals = []
            # Explicitly iterate over the true seed numbers
            for seed in self.group.seeds:
                rec = self.group.record(seed)
                err = float(np.nanmean(terminal_error_cm(
                    rollout(rec, self.paths, intervention=_iv(source, eff, rec), **_ROLLOUT_KW))))
                vals.append(err)
            self._cache[key] = np.array(vals)
        return self._cache[key]


def _mean_sd(vals: np.ndarray):
    m = vals.mean()
    s = vals.std(ddof=1) if vals.size > 1 else 0.0
    return m, s


# --- plots ---------------------------------------------------------------------------------------
def _plot_ladder(ev: _Evaluator, seeds: list[int]):
    n_seeds = len(seeds)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.array(GRID_MS, float)

    for source, label, _ci, swept in _LADDER_SOURCES:
        if swept:
            stats = [_mean_sd(ev.terminal(source, d)) for d in GRID_MS]
            mean = np.array([m for m, _ in stats]); sd = np.array([s for _, s in stats])
        else:                                              
            m, s = _mean_sd(ev.terminal(source, 0.0))
            mean = np.full_like(x, m); sd = np.full_like(x, s)
        ax.plot(x, mean, "o-", color=_colour(source), lw=2, ms=5, label=label)
        ax.fill_between(x, mean - sd, mean + sd, color=_colour(source), alpha=0.15)

    rel0, _ = _mean_sd(ev.terminal("relayed_delayed_feedback", 0.0))
    ax.axhline(rel0, ls="--", color=_colour("relayed_delayed_feedback"), lw=1.2, alpha=0.9)
    
    ax.text(x[-1], rel0 - 1.5, " naive relay at baseline (Δ=0)", 
            va="top", ha="right", fontsize=8, color=_colour("relayed_delayed_feedback"))

    ax.set_xlabel("added feedback delay Δ (ms)")
    ax.set_ylabel("terminal error (cm)")
    ax.set_title(f"Substitution ladder: the predictor supplies phase lead, (n={n_seeds})")
    
    # Place legend at the top, spread across 2 columns so it isn't too tall
    ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=True)
    
    # Add headroom to the y-axis so the legend sits safely above the highest data points
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 15)
    
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    fig.tight_layout()
    return fig


def _plot_bars(ev: _Evaluator, seeds: list[int]):
    n_seeds = len(seeds)
    labels, colours, means, sds, per_seed = [], [], [], [], []
    for label, source, d in _BARS:
        vals = ev.terminal(source, d); m, s = _mean_sd(vals)
        labels.append(label); colours.append(_colour(source))
        means.append(m); sds.append(s); per_seed.append(vals)
    means, sds = np.array(means), np.array(sds)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    xp = np.arange(len(labels))
    
    # Asymmetric error bars: lower bound clipped at 0 to prevent negative error bars
    lower_errors = np.minimum(means, sds) # Subtracts SD, but caps at 0
    upper_errors = sds
    yerr = [lower_errors, upper_errors]

    ax.bar(xp, means, yerr=yerr, color=colours, alpha=0.85, capsize=4, zorder=2)
    
    # Extract a colormap to uniquely color each seed
    cmap = plt.get_cmap("tab10") 
    
    # Plot individual seeds consistently across all bars using their true seed numbers
    for j, seed in enumerate(seeds):
        # Gather this seed's terminal error across the 8 bars
        seed_vals = [per_seed[i][j] for i in range(len(labels))]
        ax.scatter(xp, seed_vals, color=cmap(j % 10), s=35, zorder=3, alpha=0.9, 
                   edgecolors='black', linewidth=0.5, label=f"Seed {seed}")

    # Add Mean ± SD on top of the bars
    for i in range(len(labels)):                           
        ax.annotate(f"{means[i]:.1f}±{sds[i]:.1f}", (xp[i], means[i] + sds[i]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("terminal error (cm)")
    ax.set_title(f"Substitution ladder — key contrasts (mean±SD, per-seed dots, n={n_seeds})")
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.margins(y=0.15) # Leave a bit more room at the top for the text
    
    # Legend for the seeds (loc="best" typically places this in the empty top-left space)
    ax.legend(title="Network Seeds", fontsize=8, loc="best", ncol=2)

    fig.tight_layout()
    return fig


def _print_contrasts_table(ev: _Evaluator):
    print(f"\n{'Contrast / Source':<30} | {'Mean Error (cm)':<15} | {'SD (cm)'}")
    print("-" * 65)
    for label, source, d in _BARS:
        vals = ev.terminal(source, d)
        m, s = _mean_sd(vals)
        clean_label = label.replace('\n', ' ')
        print(f"{clean_label:<30} | {m:<15.2f} | {s:.2f}")
    print("-" * 65 + "\n")


def substitution_ladder_figure(cat, paths, *, save=True, show=True, print_table=True):
    apply_style()
    group = cat.canonical("dual")
    ev = _Evaluator(group, paths)
    
    # Pass the actual list of true seed numbers to the plotting functions
    ladder = _plot_ladder(ev, group.seeds)
    bars = _plot_bars(ev, group.seeds)
    
    if print_table:
        _print_contrasts_table(ev)
        
    if save:
        d = paths.aggregated_dir("R3", create=True)
        save_figure(ladder, d / "F6a_substitution_ladder", formats=("png",), close=False)
        save_figure(bars, d / "F6b_substitution_contrasts", formats=("png",), close=False)
    
    if show:
        plt.show()
        
    return ladder, bars, ev