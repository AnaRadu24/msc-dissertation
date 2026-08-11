"""Predictor-substitution manipulator (Layer 3) — swap what fills the Task Net's estimate slot.

The substitution ladder (dual only, weights frozen): the Task Net is untouched but the signal in
its state-estimate slot is replaced. This dissociates delay-COMPENSATION (a genuine forward
estimate) from mere relaying. Engine primitives:

    intact                     the trained predictor's x̂ (baseline)
    instant_feedback           true undelayed proprioception y(t)          — perfect estimator (ceiling)
    relayed_feedback           raw delayed proprioception y(t−Δ) fed in    — naive relay (no forward model)
    delayed_estimate           good x̂ delivered Δ late (output-staled)     — isolates phase/currentness
    zero                       literal zeros                                — severed, OUT-OF-DISTRIBUTION floor
    constant_predictor_output  x̂ frozen at its first in-distribution value — constant, IN-DISTRIBUTION floor

`zero` and `constant_predictor_output` are the paired floors: `zero` writes a literal zero vector
(OOD for the Task Net — muscle-length channels are never zero), whereas `constant_predictor_output`
freezes a genuine, in-distribution x̂ so any collapse cannot be blamed on distribution shift.
Comparing the two isolates 'no phase information' from 'off-manifold input'.

Pure factory: returns an Intervention that only sets estimate_source (+ a staling flag for
delayed_estimate). The engine (rollout.py) performs the actual swap.
"""
from __future__ import annotations

from ..engine.types import Intervention

SOURCES = (
    "intact", "instant_feedback", "relayed_feedback",
    "delayed_estimate", "zero", "constant_predictor_output",
)
_DESC = {
    "intact":                    "predictor estimate x̂ (trained baseline)",
    "instant_feedback":          "true undelayed proprioception y(t) (perfect estimator, ceiling)",
    "relayed_feedback":          "raw delayed proprioception y(t−Δ) fed straight in (naive relay)",
    "delayed_estimate":          "good x̂ delivered Δ late (output-staled — isolates phase/currentness)",
    "zero":                      "literal zeros (severed; OUT-OF-DISTRIBUTION floor)",
    "constant_predictor_output": "x̂ frozen at its first in-distribution value (constant, in-distribution floor)",
}


def substitute(source: str, *, estimate_delay_ms: float = 0.0) -> Intervention:
    """Fill the estimate slot with ``source``.

    For 'delayed_estimate', the intact x̂ is delivered ``estimate_delay_ms`` late (the predictor
    still runs at baseline delay, so the estimate is of the same quality — only its CURRENCY is
    degraded). This isolates the phase/lead the predictor supplies from the accuracy of x̂ itself.
    Dual only.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    overrides = {}
    if source == "delayed_estimate":
        overrides["_estimate_delay_ms"] = float(estimate_delay_ms)
    suffix = f"_{estimate_delay_ms:g}ms" if source == "delayed_estimate" else ""
    return Intervention(
        label=f"substitute_{source}{suffix}",
        description=_DESC[source],
        estimate_source=source,
        config_overrides=overrides,
    )