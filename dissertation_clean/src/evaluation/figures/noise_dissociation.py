"""F8 — noise-injection dissociation figure. PURE matplotlib.

Hold instability vs noise amplitude, one line per injection SITE, so the reader sees WHICH loop site's
corruption destabilises the hold. State-estimation sites (predictor input/output, feedback) rise;
descending-command noise stays flat — the dissociation that localises the pathology to the
forward-model pathway, a claim only a site-resolved computational model can make.

Sites are coloured by FAMILY: state-estimation (blues/greens) vs motor-command (red), so the
dissociation reads at a glance.
"""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .style import OKABE_ITO

# site -> (display label, colour, family). State-estimation family = cool colours; command = red.
_SITE_STYLE = {
    "predictor_output": ("Predictor output (x̂ — forward-model breakdown)", OKABE_ITO["blue"], "estimation"),
    "predictor_input":  ("Predictor input (afferent to cerebellum)",        OKABE_ITO["sky_blue"], "estimation"),
    "feedback":         ("Raw feedback (peripheral afferent)",              OKABE_ITO["bluish_green"], "estimation"),
    "command":          ("Descending command (motor output — control)",     OKABE_ITO["vermillion"], "command"),
}


def plot_noise_dissociation(
    sweeps: List,                       # list[NoiseSweepResult] for ONE model-seed (or averaged)
    *,
    threshold_cms: float = 10.0,
    title: str = "Where does cerebellar corruption destabilise the hold? (site dissociation)",
) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for sw in sweeps:
        label, color, family = _SITE_STYLE.get(sw.site, (sw.site, "0.5", "other"))
        ls = "-" if family == "estimation" else "--"
        ax.plot(sw.sigma, sw.hold_rms_cms, "o" + ls, color=color, lw=2, ms=5, label=label)
    ax.axhline(threshold_cms, ls=":", lw=1.2, color="0.4")
    ax.text(0.99, threshold_cms, f" onset {threshold_cms:g} cm/s", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8, color="0.4")
    ax.set_xlabel("injected noise amplitude σ")
    ax.set_ylabel("in-distribution hold-phase speed RMS (cm/s)")
    ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8.5, title="injection site")
    ax.grid(True, alpha=0.25)
    return fig

def plot_noise_dissociation_grouped(
    agg: dict,                          # output of noise_sweep_group
    *,
    threshold_cms: float = 10.0,
    n_seeds: int = 0,
    title: str = "Where does corruption destabilise the hold? (site dissociation, across seeds)",
) -> Figure:
    """F8 multi-seed: mean ± SD hold-RMS vs noise amplitude per site, error bands across seeds."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    order = ["predictor_input", "predictor_output", "feedback", "command"]
    for site in order:
        if site not in agg:
            continue
        d = agg[site]
        label, color, family = _SITE_STYLE.get(site, (site, "0.5", "other"))
        ls = "-" if family == "estimation" else "--"
        m, sd = d["mean"], np.nan_to_num(d["sd"])
        ax.plot(d["sigma"], m, "o" + ls, color=color, lw=2, ms=5, label=label)
        ax.fill_between(d["sigma"], m - sd, m + sd, color=color, alpha=0.15)
    ax.axhline(threshold_cms, ls=":", lw=1.2, color="0.4")
    ax.text(0.99, threshold_cms, f" onset {threshold_cms:g} cm/s", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8, color="0.4")
    ax.set_xlabel("injected noise amplitude σ")
    ax.set_ylabel("in-distribution hold-phase speed RMS (cm/s)")
    ttl = title + (f"  (n={n_seeds} seeds)" if n_seeds else "")
    ax.set_title(ttl)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8.5, title="injection site",
              title_fontsize=9)
    ax.grid(True, alpha=0.25)
    return fig