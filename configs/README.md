# configs/

Hydra configuration directory with a compositional layout for MARL research:

- `env` for environment family invariants
- `scenario` for task/map-specific settings
- `algo` for trainer/algorithm identity
- `model` for architecture
- `train` for optimization budgets
- `experiment` for named presets

## Recommended run entrypoint

Use the centralized runner:

```bash
python scripts/run_experiment.py
```

This keeps one command surface for Discrete SAC training and fixed-time baseline runs.

## Structure

```text
configs/
    run.yaml                     # top-level compositional config
    env/
        sumo.yaml                  # environment family defaults
        ...                        # legacy env files kept for compatibility
    scenario/
        grid5x5.yaml
        grid4x4.yaml
        arterial4x4.yaml
        cologne1.yaml
        cologne3.yaml
        cologne8.yaml
        berlin.yaml
    algo/
        discrete_sac.yaml
        fixed_time_baseline.yaml
    model/
        default.yaml
    train/
        default.yaml
        ...                        # legacy train files kept for compatibility
    logger/
        default.yaml
    runtime/
        default.yaml
    experiment/
        grid5x5.yaml
        cologne8.yaml
        fixed_time_grid5x5.yaml
```

## Why this pattern

The key design rule is separating what changes independently:

- `env`: simulator family and common mechanics
- `scenario`: map, route, time window, and task semantics
- `algo`: learning method and trainer type
- `model`: network architecture
- `train`: run budget and optimization schedule
- `experiment`: reusable named combinations

This avoids duplicating giant monolithic configs and makes sweeps natural.

## Core config fields

`run.yaml` composes groups and adds:

- `project`: project name for logging
- `seed`: random seed
- `run_name`: default run identifier
- `tags`: experiment labels

`env/sumo.yaml` contains shared SUMO defaults:

- `action_space`, `obs_space` for compatibility checks
- `common.delta_t`, `common.sumo_binary`, `common.gui`
- reward and phase constraint defaults

`scenario/*.yaml` contains scenario-level overrides only:

- `env_params.net_file`, `env_params.route_file`
- `env_params.begin_time`, `env_params.end_time`
- optional per-scenario `delta_t` and `additional_files`

`algo/*.yaml` contains trainer identity and capability constraints:

- `trainer`: `discrete_sac` / `fixed_time_baseline`
- `supports.action_space`

`train/*.yaml` contains optimization settings:

- default profile: `episodes`, `warmup`, replay and optimizer blocks

`logger/default.yaml` contains W&B configuration:

- `wandb.enabled`, `wandb.project`, `wandb.run_name`, `wandb.log_model`

`runtime/default.yaml` contains runtime infra:

- `out_dir`
- optional `gui` override

## How to run

`configs/run.yaml` sets multirun launcher to `hydra/launcher=joblib` by default,
so `-m` sweeps run in parallel.

Default parallel workers are in `configs/hydra/launcher/joblib.yaml`.
`n_jobs` is read from `HYDRA_N_JOBS` (default `3`).

Override options:

- CLI: `hydra.launcher.n_jobs=2`
- Env var: `HYDRA_N_JOBS=2`
- Auto GPU scaling helper: `python scripts/run_sweep.py ...`

Single run with defaults:

```bash
python scripts/run_experiment.py
```

Use an experiment preset:

```bash
python scripts/run_experiment.py experiment=grid5x5
python scripts/run_experiment.py experiment=fixed_time_grid5x5
```

Switch scenario and algorithm directly:

```bash
python scripts/run_experiment.py scenario=cologne8 algo=discrete_sac
```

Common parameter overrides:

```bash
# Discrete SAC trainer
python scripts/run_experiment.py \
    algo=discrete_sac \
    train.episodes=200 \
    train.batch_size=128 \
    logger.wandb.enabled=true

# Baseline with GUI
python scripts/run_experiment.py \
    algo=fixed_time_baseline \
    runtime.gui=true
```

Multi-run sweep across scenarios and seeds:

```bash
python scripts/run_experiment.py -m \
    scenario=grid4x4,cologne1,cologne8 \
    algo=discrete_sac \
    seed=1,2,3
```

Print composed config (dry run):

```bash
python scripts/run_experiment.py --cfg job
```

## Legacy entrypoints

The following utilities remain available:

- `train/evaluate.py`
- `scripts/visualize_graph_influence.py`

For new experiments, prefer `scripts/run_experiment.py`.
