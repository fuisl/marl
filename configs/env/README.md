# Scenario Run Commands (Cologne, Grid4x4 + Berlin)

This guide provides ready-to-run commands for training with
`scripts/train_gat_baseline.py` using `gat_baseline_meta_critic`.

The `net_file` and `route_file` values below are aligned with files that exist
under `nets/`.

## Download BeST Scenario to nets

```bash
python scripts/download_best_scenario.py --output-dir nets
```

By default, download statistics are recorded by the remote host. To disable
that:

```bash
python scripts/download_best_scenario.py --output-dir nets --no-record
```

## Scenarios and Files

| Scenario | Net file | Route file | Begin | End |
|---|---|---|---:|---:|
| cologne1 | `nets/cologne1/cologne1.net.xml` | `nets/cologne1/cologne1.rou.xml` | 25200 | 28800 |
| cologne3 | `nets/cologne3/cologne3.net.xml` | `nets/cologne3/cologne3.rou.xml` | 25200 | 28800 |
| cologne8 | `nets/cologne8/cologne8.net.xml` | `nets/cologne8/cologne8.rou.xml` | 25200 | 28800 |
| grid4x4 | `nets/grid4x4/grid4x4.net.xml` | `nets/grid4x4/grid4x4_1.rou.xml` | 0 | 3600 |
| berlin | `nets/berlin/berlin.net.xml` | `nets/berlin/berlin.rou.gz,nets/berlin/berlin_bus.rou.xml` | 25200 | 28800 |

## Individual Commands

### Cologne1

```bash
python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/cologne1/cologne1.net.xml \
  env.route_file=nets/cologne1/cologne1.rou.xml \
  env.begin_time=25200 \
  env.end_time=28800 \
  wandb.run_name=cologne1_meta_critic \
  runtime.out_dir=runs/cologne1_meta_critic
```

### Cologne3

```bash
python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/cologne3/cologne3.net.xml \
  env.route_file=nets/cologne3/cologne3.rou.xml \
  env.begin_time=25200 \
  env.end_time=28800 \
  wandb.run_name=cologne3_meta_critic \
  runtime.out_dir=runs/cologne3_meta_critic
```

### Cologne8

```bash
python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/cologne8/cologne8.net.xml \
  env.route_file=nets/cologne8/cologne8.rou.xml \
  env.begin_time=25200 \
  env.end_time=28800 \
  wandb.run_name=cologne8_meta_critic \
  runtime.out_dir=runs/cologne8_meta_critic
```

### Grid4x4

```bash
python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/grid4x4/grid4x4.net.xml \
  env.route_file=nets/grid4x4/grid4x4_1.rou.xml \
  env.begin_time=0 \
  env.end_time=3600 \
  wandb.run_name=grid4x4_meta_critic \
  runtime.out_dir=runs/grid4x4_meta_critic
```

### Berlin

> **Note:** The Berlin scenario loads the combined config from `configs/env/berlin.yaml`
> (28 547 nodes, 2 249 TLS junctions, ~7 h CPU sim time).  The window below
> targets the morning peak 07:00–08:00.  Use `env.begin_time=0 env.end_time=86400`
> for a full-day run.

```bash
python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env=berlin \
  wandb.run_name=berlin_meta_critic \
  runtime.out_dir=runs/berlin_meta_critic
```

## Run All Scenarios Sequentially

```bash
# Stop on first error.
set -e

python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/cologne1/cologne1.net.xml \
  env.route_file=nets/cologne1/cologne1.rou.xml \
  env.begin_time=25200 env.end_time=28800 \
  wandb.run_name=cologne1_meta_critic \
  runtime.out_dir=runs/cologne1_meta_critic

python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/cologne3/cologne3.net.xml \
  env.route_file=nets/cologne3/cologne3.rou.xml \
  env.begin_time=25200 env.end_time=28800 \
  wandb.run_name=cologne3_meta_critic \
  runtime.out_dir=runs/cologne3_meta_critic

python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/cologne8/cologne8.net.xml \
  env.route_file=nets/cologne8/cologne8.rou.xml \
  env.begin_time=25200 env.end_time=28800 \
  wandb.run_name=cologne8_meta_critic \
  runtime.out_dir=runs/cologne8_meta_critic

python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env.net_file=nets/grid4x4/grid4x4.net.xml \
  env.route_file=nets/grid4x4/grid4x4_1.rou.xml \
  env.begin_time=0 env.end_time=3600 \
  wandb.run_name=grid4x4_meta_critic \
  runtime.out_dir=runs/grid4x4_meta_critic

python scripts/train_gat_baseline.py \
  --config-name gat_baseline_meta_critic \
  env=berlin \
  wandb.run_name=berlin_meta_critic \
  runtime.out_dir=runs/berlin_meta_critic
```