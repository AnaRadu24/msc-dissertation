"""Predictor-substitution manipulator (Layer 3) — swap what fills the Task Net's estimate slot.

The substitution ladder (dual only, weights frozen): the Task Net is untouched but the signal in
its state-estimate slot is replaced — intact x̂ | zeros (floor) | raw delayed feedback y(t−Δ) |
true instant feedback y(t) (ceiling). Swept over added delay, this tests whether the predictor
does genuine delay-COMPENSATION (a forward estimate) or mere relaying: if 'delayed_feedback'
degrades with delay while 'intact' holds, the predictor supplies phase lead, not a cleaned-up past.

Pure factory: returns an Intervention setting estimate_source (the engine handles the swap).
"""
from __future__ import annotations

from ..engine.types import Intervention

SOURCES = ("intact", "zero", "delayed_feedback", "instant_feedback")
_DESC = {
    "intact":           "predictor estimate x̂ (trained baseline)",
    "zero":             "zeros — no estimate (floor)",
    "delayed_feedback": "raw delayed proprioception y(t−Δ) (relaying test)",
    "instant_feedback": "true undelayed proprioception y(t) (perfect estimator, ceiling)",
}


def substitute(source: str) -> Intervention:
    """Fill the estimate slot with ``source`` in {intact, zero, delayed_feedback, instant_feedback}.
    Dual only. Combine with ``delay(...)`` at the spec level to sweep the ladder over delay."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    return Intervention(
        label=f"substitute_{source}",
        description=_DESC[source],
        estimate_source=source,
    )