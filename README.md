# MARL Traffic Signal Control

Graph-based multi-agent reinforcement learning for SUMO traffic signal control.

## Overview

The project uses a single Hydra-based entrypoint for experiments:

```bash
python scripts/run_experiment.py
```

This command composes config groups and dispatches to one of:

- `discrete_sac` training loop
- `fixed_time_baseline` rollout

## Config Layout

The configuration is compositional, so each concern is isolated:

- `env/`: environment family defaults
- `scenario/`: map, route, and time-window settings
- `algo/`: trainer selection
- `model/`: network architecture
- `train/`: optimization schedule and budgets
- `logger/`: W&B and logging behavior
- `runtime/`: output and runtime switches
- `experiment/`: reusable presets

Top-level composition is defined in `configs/run.yaml`.

## Current Defaults

Default values come from `configs/run.yaml` + selected groups:

- `scenario=grid5x5`
- `algo=discrete_sac`
- `seed=21`
- `project=marl-traffic-gat`
- `run_name=${algo.name}_${scenario.name}_seed${seed}`

## Quick Start

```bash
# Train with defaults
python scripts/run_experiment.py

# Run a preset
python scripts/run_experiment.py experiment=grid5x5

# Train on a specific scenario
python scripts/run_experiment.py scenario=cologne8 algo=discrete_sac

# Run fixed-time baseline
python scripts/run_experiment.py algo=fixed_time_baseline
```

## Useful Overrides

```bash
# Change training budget
python scripts/run_experiment.py train.episodes=200 train.batch_size=128

# Toggle SUMO GUI
python scripts/run_experiment.py runtime.gui=true

# Override seed and output directory
python scripts/run_experiment.py seed=7 runtime.out_dir=runs/seed7
```

## Multirun Sweeps

By default, multirun uses the local Hydra Joblib composition
(`hydra/launcher=local_joblib`) in parallel (`n_jobs=3`).
You can override it per run with `hydra.launcher.n_jobs=<k>`.

Default multirun launcher is Joblib. For GPU-aware auto-scaling of
`hydra.launcher.n_jobs`, use:

```bash
python scripts/run_sweep.py \
  scenario=grid4x4,cologne1,cologne8 \
  seed=1,2,3
```

The helper script uses the currently visible GPUs (`CUDA_VISIBLE_DEVICES`) and
falls back to `1` worker if no GPU is visible.

```bash
python scripts/run_experiment.py -m \
  scenario=grid4x4,cologne1,cologne8 \
  seed=1,2,3
```

## Evaluation And Visualization

```bash
# Evaluate a trained checkpoint
python -m train.evaluate runtime.checkpoint_path=runs/<run>/best_agent.pt

# Visualize graph influence artifacts
python scripts/visualize_graph_influence.py runtime.checkpoint_path=runs/<run>/best_agent.pt
```

## Environment Setup

```bash
cp .env.example .env
```

Common variables:

- `WANDB_API_KEY`
- `WANDB_MODE`
- `LIBSUMO_AS_TRACI=1`

## Additional Docs

For full parameter reference and config examples, see `configs/README.md`.
