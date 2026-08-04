project_root/
  trained_models/            # inputs, read-only
  results/                   # all generated outputs (git-ignored)
    metrics/  figures/  aggregated/
  src/
    training/                # config.py, environments.py, networks.py, train.py
    evaluation/              #
      settings.py            #  plumbing: global params + the named time windows
      paths.py               #  plumbing: the one directory contract              
      catalogue.py           #  plumbing: discover + validate trained models    
      store.py               #  plumbing: write/read metric atoms, atoms -> DataFrame
      engine/                #  LAYER 1 — data generation & orchestration
      analyzers/             #  LAYER 2 — PURE MATHS (no matplotlib, no torch, no I/O)
      manipulators/          #  LAYER 3 — test-time interventions as pure specs
      figures/               #  LAYER 4 — PURE matplotlib/seaborn
        style.py             #    unified rcParams + colourblind palette      
  notebooks/
    explore.ipynb            #  per-variant sweep (keep every exploratory plot)
    dissertation.ipynb       #  the figure factory (reads the store)

4 layers: settings / paths / catalogue / store are four small plumbing files that connect them

Three contracts hold the whole thing together:

1) One data object flows through everything. The engine produces a RolloutResult; analysers consume it and return numbers; figures consume numbers. Nothing else is passed around. 
2) The import rule — this is the constitution. Analysers import numpy/scipy only. Figures import matplotlib/numpy only. Manipulators are inert specs — they never run a rollout. Only the engine touches torch and the environment. 
3) Outputs mirror inputs. results/{metrics,figures}/<model>/<regime>/<run_id>/<variant>/…, with the seed carried inside <run_id>.

