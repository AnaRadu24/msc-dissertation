# Predictive Compensation for Sensorimotor Delay

Codebase accompanying the dissertation. It trains recurrent neural network controllers —
a monolithic architecture and a dual Task/Predictive-Network architecture — to perform
centre-out reaching with a biomechanical arm model ([MotorNet](https://github.com/OlivierCodol/MotorNet))
under delayed sensory feedback, then evaluates both architectures' robustness to injected
delay, sensory noise, feedback ablation and predictor substitution. The evaluation battery
also tests a set of candidate mechanisms for MS-like motor dysfunction (pure delay,
forward-model breakdown, signal-dependent noise) against the resulting behavioural
signatures (reach accuracy, postural hold stability, intention-tremor crescendo).

This repository contains both the training code and the full evaluation/figure pipeline.
It is a cleaned, restructured version of the working repository (`dissertation_code`);
this snapshot is what the reported results and figures were produced from.

## Requirements

- Python 3.10 (the environment the results were produced in; other 3.10+ versions are
  likely to work but are untested)
- Dependencies listed in [requirements.txt](requirements.txt): PyTorch, MotorNet, NumPy,
  pandas, SciPy, Matplotlib, cycler

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
results/
  trained_models/<model>/<regime>/<run_id>/    inputs: config.json, *.pth, loss/trace logs
  metrics/<model>/<regime>/<run_id>/<variant>/  per-run metric atoms (metrics.json)
  figures/<model>/<regime>/<run_id>/<variant>/  per-run diagnostic figures
  aggregated/<analysis>/                        cross-model, cross-seed dissertation figures

src/
  config.py            ExperimentConfig — every training/task/physiology hyperparameter
  networks.py           MotorRNN — the single-GRU building block for both architectures
  environments.py        MotorNet reaching/hold task environments, curl-field perturbation
  train.py                the training loop (BPTT, dual-loss, early stopping, checkpointing)

  interventions/          earlier, simpler test-time hook factories (see below)

  evaluation/
    settings.py           global evaluation parameters and named rollout profiles/windows
    paths.py               the directory contract: where every input/output lives
    catalogue.py            discovers and validates trained runs, groups seeds into models
    store.py                 metric-atom read/write and cross-seed aggregation

    engine/                Layer 1 — the only layer that touches torch/MotorNet directly
    analyzers/              Layer 2 — pure numpy/scipy metrics over a rollout
    manipulators/            Layer 3 — pure factories for test-time interventions
    figures/                  Layer 4 — pure matplotlib figure functions

    orchestrate.py, sweep.py, noise_sweep.py, signature_matrix.py
                            orchestration: drive rollouts, run analysers, save metric atoms

notebooks/
  explore.ipynb           exploratory sweeps and figure generation (see note below)
  dissertation.ipynb       not populated in this snapshot (see note below)
```

`src` is not an installable package; scripts and notebooks add it to `sys.path` directly
(see the Quickstart below). Every module under `src/evaluation` imports `config`,
`environments`, `networks` and `train` as top-level modules for the same reason.

## Reproducing the results

`results/trained_models/` already contains the trained checkpoints for every
architecture/seed combination the dissertation reports (`dual_64+64u`, `mono_64u`,
`mono_128u`, seeds 42–46), so evaluation and figure generation can be run directly
against them without retraining.

### Quickstart: evaluate an existing model

```python
import sys
from pathlib import Path

ROOT = Path.cwd().parent            # run from notebooks/, or set ROOT to the project root
sys.path.insert(0, str(ROOT / "src"))

from evaluation.catalogue import Catalogue
from evaluation.paths import Paths
from evaluation.engine import rollout
from evaluation.engine.types import IDENTITY

paths = Paths(ROOT)
cat = Catalogue(paths)
print(cat.report())                  # every discovered model group and its seeds

dual = cat.canonical("dual")         # the primary multi-seed dual-architecture group
res = rollout(dual.records[0], paths, task="reach", intervention=IDENTITY)
```

### Computing metrics and figures for the dissertation results

The orchestration modules under `src/evaluation/` roll a model out under a declared
`VariantSpec` or sweep, run the appropriate analysers, and persist the result as a
metric atom (`results/metrics/.../metrics.json`) and/or a figure
(`results/figures/...` or `results/aggregated/...`):

```python
from evaluation.orchestrate import run_group
from evaluation.engine.types import BASELINE_COMPETENCE, BASELINE_HOLD
from evaluation.sweep import delay_sweep_group
from evaluation.figures import substitution_ladder_figure

run_group(dual, paths, BASELINE_COMPETENCE)   # R1: reach accuracy at the trained horizon
run_group(dual, paths, BASELINE_HOLD)         # R2: postural hold stability

sweeps = delay_sweep_group(dual, paths)       # F5: delay-induced instability, per seed
substitution_ladder_figure(cat, paths)        # F6: predictor substitution ladder
```

Each of `evaluation/orchestrate.py`, `evaluation/sweep.py`, `evaluation/noise_sweep.py`
and `evaluation/signature_matrix.py` documents, at the top of the file, which
dissertation result(s) it produces. Re-running any of these against the included
checkpoints reproduces the corresponding entries under `results/metrics/`,
`results/figures/` and `results/aggregated/` exactly, since evaluation is deterministic
given a fixed `rollout_seed` (see `evaluation/settings.py`).

### Training a model from scratch (optional)

Training is not required to reproduce the dissertation figures — the trained
checkpoints are already included — but a model can be retrained with:

```bash
cd src
python train.py
```

`train.py` is invoked directly rather than through a CLI; the configuration trained
under `__main__` (`ExperimentConfig(...)`) should be edited there, or `train_model` can
be imported and called programmatically. Output — the best-validation checkpoint(s),
loss/trace logs and a resume checkpoint — is written to
`results/<effector_type>/<architecture>_<hidden_units>u_split<ratio>/<loss_function>_<temporal_mode>_<training_regime>_<EnvClass>/<timestamp>_seed<N>/`.
An interrupted run can be continued by setting `RESUME_FROM` to that directory.

Note that this is a different layout from the one the evaluation `Catalogue` reads
(`results/trained_models/<model>/<regime>/<run_id>/`, see `evaluation/paths.py`) — the
checkpoints already included under `results/trained_models/` were organised into that
layout after training. A freshly trained run must be moved (or its `task_net.pth`,
`pred_net.pth` and `config.json` copied) into `results/trained_models/<model>/<regime>/<run_id>/`
before the `Catalogue` will discover it.

### Notebooks

- **`notebooks/explore.ipynb`** — the exploratory sweeps and figure generation used
  while developing the results, including some of the dissertation's figures. It uses
  both the current `evaluation.manipulators` API and the earlier hook factories in
  `src/interventions/` (see below).
- **`notebooks/dissertation.ipynb`** — present but empty in this snapshot; it is not
  needed to reproduce the results, since every figure it would have called is also
  reachable directly through the `evaluation` modules as shown above.

## `src/interventions/`

A small, earlier set of test-time hook factories (`get_sensory_ablation_hook`,
`get_noise_injection_hook`, `get_predictive_ablation_hook`) for freezing/zeroing a
sensory channel, injecting motor noise, and lesioning the Predictive Network's forward
estimate. These were used directly from `notebooks/explore.ipynb` to produce some of
the dissertation's earlier figures, before the channel-aware, `Intervention`-based
`evaluation/manipulators/` API existed. New evaluation code should use
`evaluation/manipulators/`, which covers the same interventions (and more: delay
injection, predictor substitution) through one consistent interface; `interventions/`
is kept because figures already in the dissertation depend on it.

## Evaluation pipeline design

The evaluation code is organised into four strict layers (see the module docstrings
under `src/evaluation/` for the full rationale):

1. **`engine/`** — the only layer that touches PyTorch or the MotorNet environment.
   Reconstructs a trained model from disk, applies a test-time `Intervention`, and
   returns a `RolloutResult`.
2. **`analyzers/`** — pure NumPy/SciPy functions from a `RolloutResult` to per-trial
   metric arrays (reach accuracy, hold-phase oscillation amplitude/frequency, intention
   tremor, command divergence latency, trajectory deviation).
3. **`manipulators/`** — pure factories that build an `Intervention` (ablation, delay,
   noise, predictor substitution) without ever running a rollout themselves.
4. **`figures/`** — pure Matplotlib functions from already-computed numbers/arrays to a
   `Figure`; never roll out a model or read a file.

`orchestrate.py`, `sweep.py`, `noise_sweep.py` and `signature_matrix.py` are the
orchestration layer that composes 1–3 and persists results; nothing outside this layer
performs I/O against `results/`.
