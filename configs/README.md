# configs/

Hydra configuration directory. All training and evaluation entrypoints are driven by configs here — no argparse, no hardcoded paths.

---

## Structure

```
configs/
├── env/
│   └── default.yaml        # SUMO environment settings (config group)
├── model/
│   └── default.yaml        # Neural network architecture (config group)
├── train/
│   └── lightning.yaml      # Lightning training hyperparameters (config group)
│
├── gat_baseline.yaml       # ← Entrypoint: scripts/train_gat_baseline.py
├── lightning_train.yaml    # ← Entrypoint: train/train.py
├── sumo_baseline.yaml      # ← Entrypoint: scripts/run_sumo_baseline.py
└── evaluate.yaml           # ← Entrypoint: train/evaluate.py
```

---

## Config groups (`env/`, `model/`, `train/`)

Shared, reusable config blocks consumed by the entrypoint configs via Hydra `defaults`.

| File | What it controls |
|---|---|
| `env/default.yaml` | SUMO net/route files, simulation window, reward function, phase constraints |
| `model/default.yaml` | GATv2Conv encoder dims, actor/critic hidden dims, `init_alpha`, `tau` |
| `train/lightning.yaml` | Seed, optimizer, SAC hyperparams, training schedule (used by Lightning trainer) |

---

## Entrypoint configs

### `gat_baseline.yaml` → `scripts/train_gat_baseline.py`

Manual SAC training loop (no Lightning). Pulls in `env: default` and `model: default`.

```bash
# Run with defaults
python scripts/train_gat_baseline.py

# Override at the CLI (Hydra syntax)
python scripts/train_gat_baseline.py train.episodes=200 train.batch_size=128
python scripts/train_gat_baseline.py train.optimizer.name=adam wandb.enabled=true
python scripts/train_gat_baseline.py env.end_time=1800 env.gui=true
```

Key top-level keys:

| Key | Description |
|---|---|
| `train.*` | Episodes, batch size, replay capacity, optimizer config |
| `train.optimizer.name` | `meta_adam` (default) or `adam` |
| `wandb.enabled` | Set `true` to log to W&B (needs `WANDB_API_KEY` in `.env`) |
| `runtime.out_dir` | Directory for checkpoints and logs |

---

### `lightning_train.yaml` → `train/train.py`

PyTorch Lightning trainer. Pulls in `env: default`, `model: default`, `train: lightning`.

```bash
# Run with defaults
python -m train.train

# Common overrides
python -m train.train train.max_epochs=50 wandb.enabled=true
python -m train.train runtime.devices=4 runtime.accelerator=gpu
python -m train.train train.optimizer.name=adam train.batch_size=512
```

Key top-level keys:

| Key | Description |
|---|---|
| `train.*` | Pulled from `train/lightning.yaml`; optimizer, SAC settings, schedule |
| `runtime.accelerator` | `auto`, `gpu`, `cpu` |
| `runtime.devices` | Number of GPUs/CPUs |
| `runtime.enable_checkpointing` | Save Lightning checkpoints |
| `wandb.*` | W&B logging (project, run name, log_model) |

---

### `sumo_baseline.yaml` → `scripts/run_sumo_baseline.py`

Runs fixed-time SUMO signals (no RL) to collect a comparison baseline. Pulls in `env: default`.

```bash
python scripts/run_sumo_baseline.py

# Override env settings
python scripts/run_sumo_baseline.py env.end_time=1800 env.gui=true
```

---

### `evaluate.yaml` → `train/evaluate.py`

Loads a trained Lightning checkpoint and evaluates it on SUMO. Pulls in `env: default` and `model: default`.

```bash
# Checkpoint path is required
python -m train.evaluate runtime.checkpoint_path=runs/lightning_train/checkpoints/last.ckpt

# Run with GUI and more episodes
python -m train.evaluate \
    runtime.checkpoint_path=runs/lightning_train/checkpoints/last.ckpt \
    runtime.episodes=10 \
    runtime.gui=true
```

---

## Secrets

All secrets (API keys, etc.) go in `.env` at the repo root — **never** in YAML. Copy `.env.example` to get started:

```bash
cp .env.example .env
# Then fill in WANDB_API_KEY
```

Variables read from `.env`:

| Variable | Purpose |
|---|---|
| `WANDB_API_KEY` | W&B authentication |
| `WANDB_MODE` | `online` / `offline` / `disabled` |
| `LIBSUMO_AS_TRACI=1` | Route libsumo through the TraCI interface |

---

## How Hydra overrides work

Any key in a YAML file can be overridden on the command line using dot-notation:

```bash
# Override a nested key
python scripts/train_gat_baseline.py train.optimizer.meta.hyper_lr=1e-6

# Override a config group entirely (swap env scenario)
python -m train.train env=my_custom_env

# Print the composed config without running (dry-run)
python scripts/train_gat_baseline.py --cfg job
```
