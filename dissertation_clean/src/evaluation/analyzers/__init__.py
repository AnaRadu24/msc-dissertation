"""Analysers (Layer 2) — PURE maths. numpy/scipy only; never matplotlib, torch, or I/O.
Each function takes a RolloutResult and returns per-trial arrays."""
from .deviation import trajectory_deviation
from .divergence import command_divergence, predicted_divergence_ms
from .oscillation import (
                          OSC_RMS_THRESHOLD_CMS,
                          dominant_frequency_hz,
                          hold_speed_rms_cms,
                          is_oscillating,
                          oscillation_metrics,
)
from .reach import path_efficiency, peak_speed_cms, reach_metrics, terminal_error_cm
from .sets import METRIC_SETS, analysers_for
from .tremor import intention_tremor_profile, tremor_metrics

__all__ = [
                          "METRIC_SETS",
                          "OSC_RMS_THRESHOLD_CMS",
                          "analysers_for",
                          "command_divergence",
                          "dominant_frequency_hz",
                          "hold_speed_rms_cms",
                          "intention_tremor_profile",
                          "is_oscillating",
                          "oscillation_metrics",
                          "path_efficiency",
                          "peak_speed_cms",
                          "predicted_divergence_ms",
                          "reach_metrics",
                          "terminal_error_cm",
                          "trajectory_deviation",
                          "tremor_metrics",
]