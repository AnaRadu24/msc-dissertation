"""Figures (Layer 4) — PURE matplotlib/seaborn. Take numbers/RolloutResult, return a Figure.
Never roll out, never load weights, never compute a stored metric. Save via style.save_figure."""
from .ablation import ablation_figures, build_deviation
from .ablation_deviation import DeviationSeries, deviation_timecourse_figure
from .crescendo import plot_crescendo
from .crescendo_diagnostic import plot_crescendo_diagnostic, plot_reach_trajectory
from .crescendo_reach import plot_crescendo_reach
from .delay_plane import DelayPlane, delay_plane_figure
from .divergence import (
    DivergencePoint,
    DivergenceTrace,
    divergence_scatter_figure,
    divergence_trace_figure,
)
from .factory import (
    FigureSpec,
    crescendo_figure,
    delay_sweep_figure,
    render_delay_series,
    render_figure,
    render_group,
)
from .hold_dynamics import (
    plot_hold_dynamics,
    plot_hold_trajectory_3d,
    plot_reach_2d_by_time,
)
from .money_plot import plot_money_plot
from .noise_dissociation import plot_noise_dissociation
from .phaseplane_composite import plot_phaseplane_composite
from .signature_matrix import plot_signature_matrix
from .style import (
    ARCH_COLORS,
    OKABE_ITO,
    QUALITATIVE,
    apply_style,
    arch_color,
    save_figure,
)
from .substitution_ladder import GRID_MS, substitution_ladder_figure

__all__ = [
    "ARCH_COLORS",
    "GRID_MS",
    "OKABE_ITO",
    "QUALITATIVE",
    "DelayPlane",
    "DeviationSeries",
    "DivergencePoint",
    "DivergenceTrace",
    "FigureSpec",
    "ablation_figures",
    "apply_style",
    "arch_color",
    "build_deviation",
    "crescendo_figure",
    "delay_plane_figure",
    "delay_sweep_figure",
    "deviation_timecourse_figure",
    "divergence_scatter_figure",
    "divergence_trace_figure",
    "plot_crescendo",
    "plot_crescendo_diagnostic",
    "plot_crescendo_reach",
    "plot_hold_dynamics",
    "plot_hold_trajectory_3d",
    "plot_money_plot",
    "plot_noise_dissociation",
    "plot_phaseplane_composite",
    "plot_reach_2d_by_time",
    "plot_reach_trajectory",
    "plot_signature_matrix",
    "render_delay_series",
    "render_figure",
    "render_group",
    "save_figure",
    "substitution_ladder_figure",
]