"""Main training entry point.

Wires together configs, Lightning module, and Trainer.

Usage::

    python -m train.train --config configs/train.yaml
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from config_utils import load_dotenv, resolve_repo_path
from train.lightning_module import TrafficMARLModule

load_dotenv()


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
# Main
# ======================================================================
@hydra.main(version_base=None, config_path="../configs", config_name="lightning_train")
def main(cfg: DictConfig) -> None:
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)

    # --- Seed ---
    seed = int(train_cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)

    # --- Module ---
    module = TrafficMARLModule(
        env_cfg=env_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    # --- Dataloaders ---
    max_epochs = int(train_cfg.get("max_epochs", 100))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 100))
    eval_interval = int(train_cfg.get("eval_interval", 10))

    train_loader = DataLoader(
        _StepDataset(steps_per_epoch), batch_size=1, shuffle=False
    )
    val_loader = DataLoader(
        _StepDataset(1), batch_size=1, shuffle=False
    )

    logger: object | bool = False
    if bool(cfg.wandb.enabled):
        save_dir = resolve_repo_path(cfg.runtime.out_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(
            project=str(cfg.wandb.project),
            name=cfg.wandb.run_name,
            save_dir=str(save_dir),
            log_model=bool(cfg.wandb.log_model),
        )
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    elif bool(cfg.runtime.csv_logger):
        save_dir = resolve_repo_path(cfg.runtime.out_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = CSVLogger(save_dir=str(save_dir), name="lightning_train")

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=eval_interval,
        enable_checkpointing=bool(cfg.runtime.enable_checkpointing),
        logger=logger,
        log_every_n_steps=int(cfg.runtime.log_every_n_steps),
        accelerator=str(cfg.runtime.accelerator),
        devices=int(cfg.runtime.devices),
        deterministic=bool(cfg.runtime.deterministic),
        default_root_dir=str(resolve_repo_path(cfg.runtime.out_dir)),
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
