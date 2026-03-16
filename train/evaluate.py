"""Evaluation script — run a trained raw-loop agent on SUMO and report metrics.

Usage::

    python -m train.evaluate runtime.checkpoint_path=runs/gat_baseline/best_agent.pt
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

    env = TrafficSignalEnv(**env_cfg)
    td0 = env.reset()
    obs_dim = int(td0.get("graph_observation", td0["agents", "observation"]).shape[-1])
    num_actions = int(env.num_actions)

    agent = MARLDiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        **model_cfg,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

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

    env_common = OmegaConf.to_container(cfg.env.common, resolve=True)
    scenario_params = OmegaConf.to_container(cfg.scenario.env_params, resolve=True)
    if not isinstance(env_common, dict) or not isinstance(scenario_params, dict):
        raise ValueError("Invalid env/scenario config for evaluation.")
    merged_env = {**env_common, **scenario_params}

    evaluate(
        checkpoint_path=str(resolve_repo_path(checkpoint_path)),
        env_cfg=merged_env,
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        n_episodes=int(cfg.runtime.episodes),
        device=str(cfg.runtime.device),
        gui=bool(cfg.runtime.gui),
    )


if __name__ == "__main__":
    main()
