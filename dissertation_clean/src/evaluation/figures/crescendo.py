"""R4 — intention-tremor crescendo figure. PURE matplotlib.

Intention tremor is defined clinically by a CRESCENDO: the oscillation amplitude GROWS as the hand
approaches the target (unlike postural tremor, which is distance-independent). This figure plots the
lateral-oscillation RMS binned by distance-to-target, overlaid for several injected delays, so the
claim is COMPARATIVE: the intact/low-delay condition is flat (no crescendo, index ~1) and the
past-Δc condition RISES toward the target (index > 1). That contrast is what earns the word
"intention" — a single rising curve alone would not.

HONEST SCOPE: this figure supports the SPATIAL signature (crescendo on approach) only. The model's
oscillation frequency (~0.8 Hz) does NOT match clinical intention tremor (3-5 Hz); that mismatch is a
stated limitation, not shown here. Do not let this figure imply a frequency match.

Reads intention_tremor_profile (analyzers.tremor) outputs — pure plotting, no rollout.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..analyzers.tremor import intention_tremor_profile
from ..engine.types import RolloutResult
from .style import QUALITATIVE


def plot_crescendo(
    profiles_by_label: dict,
    *,
    title: str = "Intention-tremor signature: does the oscillation crescendo on approach?",
) -> Figure:
    """``profiles_by_label`` maps a condition label -> the dict returned by intention_tremor_profile.
    Each condition is drawn as amplitude (lateral-oscillation RMS) vs distance-to-target, mean ± SEM
    across trials, with its scalar intention index in the legend. X-axis is inverted so 'approach'
    reads left->right (far -> near)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for i, (label, prof) in enumerate(profiles_by_label.items()):
        centers = prof["bin_centers_cm"]
        rms = prof["rms_by_bin_cm"]                              # [nbins, B]
        mean = np.nanmean(rms, axis=1)
        n = np.maximum(np.sum(~np.isnan(rms), axis=1), 1)
        sem = np.nanstd(rms, axis=1) / np.sqrt(n)
        idx = float(np.nanmean(prof["intention_index"]))
        color = QUALITATIVE[i % len(QUALITATIVE)]
        ax.plot(centers, mean, "o-", color=color, lw=1.8,
                label=f"{label}  (index {idx:.2f})")
        ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.15)
    ax.invert_xaxis()                                           # far -> near reads left -> right
    ax.set_xlabel("distance to target (cm)     approach →")
    ax.set_ylabel("lateral oscillation RMS (cm)")
    ax.set_title(title)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9,
              title="index > 1 = crescendo = intention-like")
    ax.grid(True, alpha=0.25)
    return fig