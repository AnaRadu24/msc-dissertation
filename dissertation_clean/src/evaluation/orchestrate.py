"""Orchestrator — the join between the engine, the analysers and the store.

run_variant  : one trained model-seed × one VariantSpec -> one MetricAtom on disk.
run_group    : every seed of a ModelGroup under one spec -> a list of atoms.
run_specs    : one model × several specs (e.g. the baseline's competence + hold).

The ONLY orchestration logic lives here: resolve the profile to (duration, batch), roll out
once (option a — one profile per spec), run the declared metric set, summarise ACROSS TRIALS,
and persist. It records the full provenance (intervention, task, profile, metric set) into the
atom so a stored number always traces to exactly how it was produced. No maths, no plotting.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .analyzers.oscillation import hold_speed_rms_cms
from .analyzers.reach import terminal_error_cm
from .analyzers.sets import analysers_for
from .catalogue import ModelGroup, ModelRecord
from .engine import obs_layout, rollout
from .engine.types import IDENTITY, Intervention, VariantSpec
from .figures.ablation import (
    AblationCascade,
    AblationSeries,
    LoopSpeed,
    LoopSpeedCausal,
    LoopSpeedRobust,
)
from .figures.delay_plane import DelayPlane, delay_plane_figure
from .figures.divergence import DivergencePoint, DivergenceTrace
from .figures.hold_dynamics import plot_reach_2d_by_time
from .figures.phaseplane_composite import plot_phaseplane_composite
from .manipulators.ablation import ablate
from .paths import Paths
from .settings import DEFAULT, Settings
from .store import MetricAtom


def _summarise_across_trials(per_trial: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Collapse each metric's per-trial [B] array to {mean, std, n} across trials. NaNs are
    ignored (nanmean/nanstd) so a metric undefined on some trials (e.g. an unresolvable
    frequency) does not poison the summary; n counts the finite trials."""
    out = {}
    for name, values in per_trial.items():
        v = np.asarray(values, dtype=float)
        finite = np.isfinite(v)
        n = int(finite.sum())
        out[name] = {
            "mean": float(np.nanmean(v)) if n else float("nan"),
            "std":  float(np.nanstd(v)) if n else float("nan"),
            "n":    n,
        }
    return out


def run_variant(record: ModelRecord, paths: Paths, spec: VariantSpec,
                settings: Settings = DEFAULT, *, save: bool = True) -> MetricAtom:
    """Roll ``record`` out under ``spec`` once, compute its metric set, write the atom."""
    profile = settings.profile(spec.profile)

    res = rollout(
        record, paths,
        task=spec.task,
        duration_s=profile.duration_s,          # None => trained horizon
        batch_size=profile.batch_size,
        intervention=spec.intervention,
        seed=settings.rollout_seed,
        device=settings.device,
    )

    # run the declared analysers; assert no metric-name collisions across them
    per_trial: Dict[str, np.ndarray] = {}
    for analyser in analysers_for(spec.metrics):
        produced = analyser(res)
        clash = set(produced) & set(per_trial)
        assert not clash, f"metric-set '{spec.metrics}' has colliding keys {clash}"
        per_trial.update(produced)

    atom = MetricAtom(
        model=record.label, regime=record.regime, run_id=record.key.run_id,
        seed=record.seed, architecture=record.architecture,
        hidden_units=record.hidden_units,
        variant=spec.name, description=spec.intervention.description,
        config_overrides=dict(spec.intervention.config_overrides),
        task=spec.task, profile=spec.profile,
        duration_s=res.duration_s, batch_size=profile.batch_size,
        rollout_seed=settings.rollout_seed, n_trials=res.n_trials,
        metrics=_summarise_across_trials(per_trial),
    )
    if save:
        path = paths.metrics_atom(record.key, spec.name, create=True)
        atom.save(path)
        print(f"[atom] {record.label} seed{record.seed} · {spec.name} -> {path}")
    return atom


def run_group(group: ModelGroup, paths: Paths, spec: VariantSpec,
              settings: Settings = DEFAULT, *, save: bool = True) -> List[MetricAtom]:
    """Run one spec across every seed of a model group (the unit that gets aggregated over)."""
    return [run_variant(g, paths, spec, settings, save=save) for g in group.records]


def run_specs(record: ModelRecord, paths: Paths, specs: List[VariantSpec],
              settings: Settings = DEFAULT, *, save: bool = True) -> List[MetricAtom]:
    """Run several specs on one model (e.g. the baseline's competence + hold measurements)."""
    return [run_variant(record, paths, spec, settings, save=save) for spec in specs]

# R1 ablation: rolls each (model, condition) out once; reads BOTH metrics off that single rollout.
_KW   = dict(task="reach", duration_s=1.0, batch_size=256, seed=42)
_HOLD_START = 0.6      # hold window is [0.6 s, rollout end]; end is implicit in the analyser

def _interventions(rec):
    layout = obs_layout(rec)          # fetched once per model; ablate() needs it
    return {"intact": IDENTITY,
            "vision": ablate("vision", layout),
            "prop":   ablate("prop",   layout),
            "both":   ablate("both",   layout)}

def _both_metrics(rec, paths, iv):
    res = rollout(rec, paths, intervention=iv, **_KW)
    te  = float(np.nanmean(terminal_error_cm(res)))
    rms = float(np.nanmean(hold_speed_rms_cms(res, hold_start_s=_HOLD_START)))   # explicit start!
    return te, rms

def _series(group, paths):            # one architecture -> AblationSeries (terminal only)
    vals = {c: [] for c in ("intact", "vision", "prop", "both")}
    for rec in group.records:
        ivs = _interventions(rec)
        for c in vals:
            vals[c].append(_both_metrics(rec, paths, ivs[c])[0])
    return AblationSeries(tuple(group.seeds), {c: np.array(v) for c, v in vals.items()})

def build_cascade(cat, paths):
    return AblationCascade(dual=_series(cat.canonical("dual"), paths),
                           mono=_series(cat.canonical("mono"), paths))

def build_loopspeed(cat, paths, causal_records):
    # robust: dual n=5, prop & vision freeze, BOTH metrics
    g = cat.canonical("dual")
    tp, tv, hp, hv = [], [], [], []
    for rec in g.records:
        ivs = _interventions(rec)
        te_p, rms_p = _both_metrics(rec, paths, ivs["prop"])
        te_v, rms_v = _both_metrics(rec, paths, ivs["vision"])
        tp.append(te_p); tv.append(te_v); hp.append(rms_p); hv.append(rms_v)
    robust = LoopSpeedRobust(tuple(g.seeds), np.array(tp), np.array(tv),
                             np.array(hp), np.array(hv))
    # causal: single seed per delay assignment (the 3 records are supplied in order)
    order = ("baseline", "matched", "swapped")
    tp, tv, hp, hv = [], [], [], []
    for cfg in order:
        rec = causal_records[cfg]
        ivs = _interventions(rec)
        te_p, rms_p = _both_metrics(rec, paths, ivs["prop"])
        te_v, rms_v = _both_metrics(rec, paths, ivs["vision"])
        tp.append(te_p); tv.append(te_v); hp.append(rms_p); hv.append(rms_v)
    causal = LoopSpeedCausal(order, np.array(tp), np.array(tv),
                             np.array(hp), np.array(hv))
    return LoopSpeed(robust, causal)

def _inject(prop_ms, vision_ms):
    """Independent additive delay per channel via the engine's _delay_add_* sentinels.
    (Set both directly — _resolved_config resolves each against that model's trained base.)"""
    return Intervention(label=f"delay_p{prop_ms:g}_v{vision_ms:g}",
                        description=f"+{prop_ms} ms prop, +{vision_ms} ms vision",
                        config_overrides={"_delay_add_prop_ms": float(prop_ms),
                                          "_delay_add_vision_ms": float(vision_ms)})

def build_delay_plane(group, paths, prop_grid, vision_grid, *, marks=()):
    """Across-seed mean terminal error at t=1 s over the injected-delay grid, for ONE
    architecture group. Keeps per-seed values so panel cells can be checked."""
    P, V, S = len(prop_grid), len(vision_grid), len(group.seeds)
    per = np.full((P, V, S), np.nan)
    for i, p in enumerate(prop_grid):
        for j, v in enumerate(vision_grid):
            for s, rec in enumerate(group.records):          # records are seed-sorted -> [...,0] = seed 42
                res = rollout(rec, paths, intervention=_inject(p, v),
                              task="reach", duration_s=1.0, batch_size=256, seed=42)
                per[i, j, s] = np.nanmean(terminal_error_cm(res))
    return DelayPlane(np.asarray(prop_grid, float), np.asarray(vision_grid, float),
                      terminal_cm=np.nanmean(per, axis=2), terminal_cm_per_seed=per,
                      n_seeds=S, marks=marks)

def check_seed42_representative(plane, pairs):
    """Print seed-42 terminal vs across-seed mean at each (prop, vision) cell shown as a panel."""
    for p, v in pairs:
        i = int(np.argmin(np.abs(plane.prop_ms - p)))
        j = int(np.argmin(np.abs(plane.vision_ms - v)))
        s42, m = plane.terminal_cm_per_seed[i, j, 0], plane.terminal_cm[i, j]
        flag = "  <-- check" if abs(s42 - m) > 0.5 * max(m, 1.0) else ""
        print(f"(prop {p:>4}, vis {v:>4}): seed42={s42:6.2f} cm   mean={m:6.2f} cm{flag}")

def delay_plane_panels(group, paths, pairs, *, duration_s=6.0, seed_for_panels=42):
    """Trajectory + phase-plane composite at each cell, seed 42, over `duration_s`
    (6 s to let the hold reveal a limit cycle, exactly as R2). The heatmap metric stays
    the 1 s terminal error; these panels only CHARACTERISE the same instability."""
    rec = group.record(seed_for_panels)
    figs = {}
    for p, v in pairs:
        res = rollout(rec, paths, intervention=_inject(p, v),
                      task="reach", duration_s=duration_s, batch_size=256, seed=42)
        tag = f"{group.architecture}_p{p:g}_v{v:g}"
        figs[f"traj_{tag}"] = plot_reach_2d_by_time(res, direction=0, n_trials=3,
                                  title=f"{group.architecture} · +{p} ms prop, +{v} ms vision")
        figs[f"composite_{tag}"] = plot_phaseplane_composite(res, trial=0,
                                  title=f"{group.architecture} · +{p} ms prop, +{v} ms vision")
    return figs


def build_divergence(cat, paths, causal_records):
    """Computes the exact millisecond the network issues a corrective command.

    Method: rolls out the identical network, seed, and targets in both an unperturbed
    environment (reach) and a perturbed one (curl). The muscle excitations remain
    bit-identical until the exact moment the sensory feedback informing the network of
    the perturbation clears the delay lines.
    """
    points = []
    trace_out = None

    # A short rollout (1.0 s) suffices because divergence happens early in the reach
    _KW = dict(duration_s=1.0, batch_size=256, seed=42)

    for label, rec in causal_records.items():
        base_p = rec.config.proprioception_delay
        base_v = rec.config.vision_delay

        # Theoretical minimum available delay for each ablation condition
        conditions = {
            "intact": min(base_p, base_v),
            "vision": base_p,  # Vision severed -> forced to wait for slower proprioception
            "prop": base_v,    # Proprioception severed -> forced to wait for slower vision
            "both": np.inf     # Totally severed -> open loop (should never diverge)
        }

        layout = obs_layout(rec)

        for ab_name, predicted_delay in conditions.items():
            intervention = IDENTITY if ab_name == "intact" else ablate(ab_name, layout)

            unpert = rollout(rec, paths, task="reach", intervention=intervention, **_KW)
            pert = rollout(rec, paths, task="curl", intervention=intervention, **_KW)

            # Millisecond at which the muscle commands diverge; a difference > 1e-5 filters
            # out floating-point noise
            diff = np.abs(pert.motor_commands - unpert.motor_commands)
            max_diff_across_batch = diff.max(axis=(1, 2))  # Collapse batch and muscles -> (T,)
            
            diverge_idx = np.where(max_diff_across_batch > 1e-5)[0]
            
            if len(diverge_idx) > 0:
                measured_ms = float(diverge_idx[0] * unpert.dt * 1000)
            else:
                measured_ms = np.nan  # Network never reacted (open loop)
                
            points.append(DivergencePoint(
                config=label,
                predicted_ms=int(predicted_delay) if predicted_delay != np.inf else 300, 
                measured_ms=measured_ms,
                ablation=ab_name
            ))
            
            # Cache ONE specific trace for the Appendix plot (Canonical Intact model)
            if trace_out is None and ab_name == "intact" and label == "baseline":
                trace_out = DivergenceTrace(
                    t_ms=np.arange(len(max_diff_across_batch)) * unpert.dt * 1000,
                    excitation_perturbed=pert.motor_commands[:, 0, :],     # Plot trial 0
                    excitation_unperturbed=unpert.motor_commands[:, 0, :], # Plot trial 0
                    onset_ms=100.0, # Standard movement onset time
                    predicted_ms=predicted_delay,
                    measured_ms=measured_ms,
                    max_abs_delta=max_diff_across_batch,
                    muscle_labels=["SF", "SE", "BF", "BE", "EF", "EE"]
                )

    return points, trace_out