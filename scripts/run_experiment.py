"""Centralized experiment runner for MARL traffic research.

This script is the single entrypoint for training and baseline experiments.
It composes config groups (env/scenario/algo/model/train/logger/runtime)
using Hydra, validates compatibility, and dispatches to the selected trainer.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py experiment=grid5x5
    python scripts/run_experiment.py scenario=cologne8 algo=discrete_sac
    python scripts/run_experiment.py -m scenario=grid5x5,cologne1 algo=discrete_sac seed=1,2
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_utils import load_dotenv, resolve_repo_path
from scripts.run_sumo_baseline import run_baseline
from scripts.train_gat_baseline import train as run_training_loop

load_dotenv()


def _as_plain(cfg: Any) -> Any:
    return OmegaConf.to_container(cfg, resolve=True)


def _merge_env_cfg(cfg: DictConfig) -> dict[str, Any]:
    env_common = _as_plain(cfg.env.common)
    scenario_params = _as_plain(cfg.scenario.env_params)
    if not isinstance(env_common, dict) or not isinstance(scenario_params, dict):
        raise ValueError("Invalid env/scenario config. Expected mapping objects.")

    merged = {**env_common, **scenario_params}
    if cfg.runtime.gui is not None:
        merged["gui"] = bool(cfg.runtime.gui)
    return merged


def _validate_cfg(cfg: DictConfig) -> None:
    env_action_space = str(cfg.env.action_space)
    supported = list(cfg.algo.supports.action_space)
    if env_action_space not in supported:
        raise ValueError(
            f"Algorithm '{cfg.algo.name}' does not support action space "
            f"'{env_action_space}'. Supported: {supported}."
        )


def _run_training(cfg: DictConfig, env_cfg: dict[str, Any]) -> None:
    run_cfg = {
        "env": env_cfg,
        "model": _as_plain(cfg.model),
        "train": _as_plain(cfg.train),
        "wandb": _as_plain(cfg.logger.wandb),
        "runtime": {
            "out_dir": cfg.runtime.out_dir,
        },
    }
    run_cfg["train"]["seed"] = int(cfg.seed)
    run_cfg["wandb"]["run_name"] = str(run_cfg["wandb"].get("run_name") or cfg.run_name)
    run_training_loop(OmegaConf.create(run_cfg))


@hydra.main(version_base=None, config_path="../configs", config_name="run")
def main(cfg: DictConfig) -> None:
    _validate_cfg(cfg)
    env_cfg = _merge_env_cfg(cfg)

    trainer = str(cfg.algo.trainer)
    if trainer == "discrete_sac":
        _run_training(cfg, env_cfg)
        return
    if trainer == "fixed_time_baseline":
        run_baseline(env_cfg)
        return

    raise ValueError(f"Unknown algo.trainer '{trainer}'.")


if __name__ == "__main__":
    main()
