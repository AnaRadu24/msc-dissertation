"""R4 payoff figure — the hypothesis × signature matrix as an annotated heatmap. PURE matplotlib."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_signature_matrix(text_df, num_df, *,
                          title="MS hypotheses produce dissociable motor signatures") -> Figure:
    """text_df: mean±SD strings; num_df: numeric means (for colour). Rows=hypotheses, cols=signatures.
    Each column is colour-normalised independently (they have different units)."""
    col_labels = {"terminal_error_cm": "Terminal\nerror (cm)",
                  "hold_rms_cms": "Hold\ninstability (cm/s)",
                  "crescendo_index": "Crescendo\nindex"}
    cols = list(num_df.columns)
    rows = list(num_df.index)
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.7 * len(rows) + 2))

    # per-column normalised colour (each signature has its own scale)
    norm = num_df.copy()
    for c in cols:
        v = num_df[c].values.astype(float)
        rng = np.nanmax(v) - np.nanmin(v)
        norm[c] = (v - np.nanmin(v)) / rng if rng > 0 else 0.0
    ax.imshow(norm.values, cmap="OrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([col_labels.get(c, c) for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j, c in enumerate(cols):
            ax.text(j, i, text_df.iloc[i][c], ha="center", va="center", fontsize=9)
    ax.set_title(title, pad=12)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="w", lw=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    return fig