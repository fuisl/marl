"""Evaluation script — run a trained agent on SUMO and report traffic metrics.

Usage::

    python -m train.evaluate --checkpoint path/to/checkpoint.ckpt --config configs/train.yaml
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from config_utils import load_dotenv, resolve_repo_path
from marl_env.sumo_env import TrafficSignalEnv
from models.marl_discrete_sac import MARLDiscreteSAC
from rl.rollout import RolloutWorker
from train.lightning_module import TrafficMARLModule

load_dotenv()


def evaluate(
    checkpoint_path: str,
    env_cfg: dict,
    model_cfg: dict,
    *,
    n_episodes: int = 5,
    device: str = "cpu",
    gui: bool = False,
) -> list[dict[str, float]]:
    """Load a checkpoint and evaluate for ``n_episodes``.

    Returns
    -------
    list of dicts
        Per-episode metrics (episode_return, episode_length, ...).
    """
    # Override GUI setting for visual evaluation
    env_cfg = {**env_cfg, "gui": gui}

    module = TrafficMARLModule.load_from_checkpoint(
        checkpoint_path,
        env_cfg=env_cfg,
        model_cfg=model_cfg,
    )
    agent: MARLDiscreteSAC = module.agent  # type: ignore[assignment]
    agent = agent.to(device)
    agent.eval()

    env = TrafficSignalEnv(**env_cfg)
    worker = RolloutWorker(env=env, agent=agent, device=device)

    all_metrics: list[dict[str, float]] = []
    for ep in range(n_episodes):
        _, info = worker.collect_episode(deterministic=True)
        print(f"Episode {ep + 1}/{n_episodes}: {info}")
        all_metrics.append(info)

    env.close()

    # Summary
    avg_return = sum(m["episode_return"] for m in all_metrics) / n_episodes
    avg_length = sum(m["episode_length"] for m in all_metrics) / n_episodes
    print(f"\n--- Evaluation summary ({n_episodes} episodes) ---")
    print(f"  Avg return:  {avg_return:.2f}")
    print(f"  Avg length:  {avg_length:.1f}")

    return all_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.runtime.checkpoint_path
    if checkpoint_path in (None, ""):
        raise ValueError("Set runtime.checkpoint_path in config or via Hydra override.")

    evaluate(
        checkpoint_path=str(resolve_repo_path(checkpoint_path)),
        env_cfg=OmegaConf.to_container(cfg.env, resolve=True),
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        n_episodes=int(cfg.runtime.episodes),
        device=str(cfg.runtime.device),
        gui=bool(cfg.runtime.gui),
    )


if __name__ == "__main__":
    main()
