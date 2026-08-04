"""Delay manipulator (Layer 3) — inject extra feedback delay at TEST time.

A delay is a pure CONFIG override: proprioception_delay / vision_delay are read only when the env
is built, so overriding them evaluates the SAME trained weights under a longer loop delay. The R2
"same controller, more delay" claim depends on it being an override, never a retrain.

A DelayMode maps ONE scalar sweep parameter to a (Δprop, Δvision) increment pair. Every delay
analysis is a sweep over some mode, so this one abstraction serves them all:

    additive_both     +x ms to both channels           (R2 money plot — simplest, one knob)
    additive_prop     +x ms to proprioception only      (§2.5 mechanism — isolate the channel)
    additive_vision   +x ms to vision only              (§2.5 mechanism)
    multiplicative    scale BOTH base delays by factor  (R4 demyelination — slowing ∝ pathway
                      x=k>=1 (so +base*(k-1) each)        length, so the longer visual pathway
                                                          gains more absolute delay)

The increment is ADDITIVE on each model's own trained base delay (resolved in the engine via the
_delay_add_* sentinels), so a given sweep value means the same physical thing across the
swapped/matched variants whose base delays differ. Pure factory: returns an Intervention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..engine.types import Intervention


@dataclass(frozen=True)
class DelayMode:
    """Maps a scalar sweep parameter -> (added_prop_ms, added_vision_ms) increment.

    ``increments(x, base_prop, base_vision)`` returns the ms to ADD to each channel at sweep value
    ``x``. ``unit`` / ``param_label`` describe the sweep axis (for figures), so nothing downstream
    hardcodes 'ms' vs the multiplicative factor 'k'. ``x_is_ms`` says whether ``x`` is already an
    absolute-ms grid value (additive modes) or a dimensionless factor (multiplicative)."""
    name: str
    unit: str                                     # axis unit for figures, e.g. "ms" or "×"
    param_label: str                              # axis label, e.g. "added delay (ms)"
    x_is_ms: bool
    increments: Callable[[float, float, float], tuple[float, float]]


def _round10(v: float) -> float:
    """MotorNet requires delays on the 10 ms timestep grid; round increments to it."""
    return round(v / 10.0) * 10.0


DELAY_MODES = {
    "additive_both":   DelayMode("additive_both", "ms", "added delay (ms)", True,
                                 lambda x, bp, bv: (x, x)),
    "additive_prop":   DelayMode("additive_prop", "ms", "added proprioception delay (ms)", True,
                                 lambda x, bp, bv: (x, 0.0)),
    "additive_vision": DelayMode("additive_vision", "ms", "added vision delay (ms)", True,
                                 lambda x, bp, bv: (0.0, x)),
    # x = severity factor k >= 1; added = base*(k-1), rounded to the timestep grid.
    "multiplicative":  DelayMode("multiplicative", "×", "conduction-slowing factor k", False,
                                 lambda k, bp, bv: (_round10(bp * (k - 1.0)),
                                                    _round10(bv * (k - 1.0)))),
}


def delay(x: float, *, mode: str = "additive_both",
          base_prop_ms: float, base_vision_ms: float) -> Intervention:
    """Build a delay Intervention at sweep value ``x`` under ``mode``.

    Needs the model's trained base delays (from record.config) because multiplicative slowing and
    the additive-on-base semantics both depend on them. The sweep driver passes these in; a
    notebook rarely calls this directly. x must place both increments on the 10 ms grid.
    """
    if mode not in DELAY_MODES:
        raise ValueError(f"mode must be one of {sorted(DELAY_MODES)}, got {mode!r}")
    m = DELAY_MODES[mode]
    add_prop, add_vision = m.increments(float(x), float(base_prop_ms), float(base_vision_ms))
    for nm, val in (("prop", add_prop), ("vision", add_vision)):
        if val % 10 != 0:
            raise ValueError(f"{mode}: added {nm} delay {val} ms is not a multiple of the 10 ms "
                             f"timestep; choose sweep values that land on the grid.")
    overrides = {}
    if add_prop:
        overrides["_delay_add_prop_ms"] = add_prop
    if add_vision:
        overrides["_delay_add_vision_ms"] = add_vision
    xtag = f"{x:g}{m.unit}" if not m.x_is_ms else f"+{x:g}ms"
    return Intervention(
        label=f"delay_{mode}_{xtag}",
        description=(f"{mode} delay sweep at {x:g}{m.unit}: +{add_prop:g} ms prop, "
                    f"+{add_vision:g} ms vision (weights frozen)"),
        config_overrides=overrides,
    )