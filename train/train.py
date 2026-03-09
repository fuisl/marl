"""Main training entry point.

Wires together configs, Lightning module, and Trainer.

Usage::

    python -m train.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import yaml
from torch.utils.data import DataLoader, Dataset

from train.lightning_module import TrafficMARLModule


# ======================================================================
# Dummy dataset — Lightning requires a dataloader even in manual mode.
# Each "sample" is just an integer step index.
# ======================================================================
class _StepDataset(Dataset):
    """Dummy dataset that yields step indices."""

    def __init__(self, n_steps: int) -> None:
        self.n_steps = n_steps

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int) -> int:
        return idx


# ======================================================================
# Config loading
# ======================================================================
def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ======================================================================
# Main
# ======================================================================
def main(config_path: str | None = None) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, default="configs/train.yaml",
            help="Path to training config YAML.",
        )
        args = parser.parse_args()
        config_path = args.config

    cfg = load_config(config_path)

    env_cfg = cfg.get("env", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    # --- Seed ---
    seed = train_cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)

    # --- Module ---
    module = TrafficMARLModule(
        env_cfg=env_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    # --- Dataloaders ---
    max_epochs = train_cfg.get("max_epochs", 100)
    steps_per_epoch = train_cfg.get("steps_per_epoch", 100)
    eval_interval = train_cfg.get("eval_interval", 10)

    train_loader = DataLoader(
        _StepDataset(steps_per_epoch), batch_size=1, shuffle=False
    )
    val_loader = DataLoader(
        _StepDataset(1), batch_size=1, shuffle=False
    )

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=eval_interval,
        enable_checkpointing=True,
        logger=True,  # uses default TensorBoard logger
        log_every_n_steps=1,
        accelerator="auto",
        devices=1,
        deterministic=False,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
