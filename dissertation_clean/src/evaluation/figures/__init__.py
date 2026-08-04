"""Figures (Layer 4) — PURE matplotlib/seaborn. Take numbers/RolloutResult, return a Figure.
Never roll out, never load weights, never compute a stored metric. Save via style.save_figure."""
from .crescendo import plot_crescendo
from .factory import (
           FigureSpec,
           crescendo_figure,
           delay_sweep_figure,
           render_delay_series,
           render_figure,
           render_group,
)
from .hold_dynamics import plot_hold_dynamics, plot_hold_trajectory_3d
from .money_plot import plot_money_plot
from .noise_dissociation import plot_noise_dissociation
from .signature_matrix import plot_signature_matrix
from .style import (
           ARCH_COLORS,
           OKABE_ITO,
           QUALITATIVE,
           apply_style,
           arch_color,
           save_figure,
)

__all__ = [
           "ARCH_COLORS",
           "OKABE_ITO",
           "QUALITATIVE",
           "FigureSpec",
           "apply_style",
           "arch_color",
           "crescendo_figure",
           "delay_sweep_figure",
           "plot_crescendo",
           "plot_hold_dynamics",
           "plot_hold_trajectory_3d",
           "plot_money_plot",
           "plot_noise_dissociation",
           "plot_signature_matrix",
           "render_delay_series",
           "render_figure",
           "render_group",
           "save_figure",
]