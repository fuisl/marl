# MARL Traffic Signal Control

Graph-based multi-agent reinforcement learning for SUMO traffic signal control.

## Centralized Experiment Runner

Use one command surface for training and baselines:

```bash
python scripts/run_experiment.py
```

This runner composes Hydra config groups and dispatches to:

- Discrete SAC trainer
- fixed-time SUMO baseline

## Config Pattern

Configs follow a compositional research layout:

- `env/`: environment family defaults
- `scenario/`: map/task-specific parameters
- `algo/`: algorithm/trainer selection
- `model/`: network architecture
- `train/`: optimization schedule and run budget
- `experiment/`: reusable named presets

Top-level composition is in `configs/run.yaml`.

## Quick Start

```bash
# Discrete SAC on grid5x5 (default)
python scripts/run_experiment.py

# Use a named preset
python scripts/run_experiment.py experiment=grid5x5

# Override scenario + algorithm
python scripts/run_experiment.py scenario=cologne8 algo=discrete_sac

# Fixed-time baseline
python scripts/run_experiment.py algo=fixed_time_baseline
```

## Sweeps

```bash
python scripts/run_experiment.py -m \
  scenario=grid4x4,cologne1,cologne8 \
  algo=discrete_sac \
  seed=1,2,3
```

## Legacy Entrypoints

The older scripts are still available for compatibility:

- `scripts/train_gat_baseline.py`
- `scripts/run_sumo_baseline.py`
- `train/evaluate.py`

For new work, prefer `scripts/run_experiment.py`.

## Environment Setup

```bash
cp .env.example .env
```

Typical variables:

- `WANDB_API_KEY`
- `WANDB_MODE`
- `LIBSUMO_AS_TRACI=1`

## More Details

See `configs/README.md` for full parameter-level documentation and examples.
