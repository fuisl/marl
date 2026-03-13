"""Run SUMO with its default fixed-time traffic signal programs (baseline).

Collects the same traffic metrics that the RL agent will be evaluated on,
so you have a fair comparison baseline.

Usage::

    python scripts/run_sumo_baseline.py --config configs/env.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_utils import load_dotenv, maybe_to_container

from marl_env.sumo_env import TrafficSignalEnv

load_dotenv()


def run_baseline(env_cfg: dict) -> dict[str, float]:
    """Run one full episode with SUMO's default signal programs.

    No RL actions are applied — the environment just steps forward
    with whatever fixed-time program SUMO loaded from the network file.
    """
    env = TrafficSignalEnv(**env_cfg)
    td = env.reset()

    total_reward = 0.0
    steps = 0
    n_agents = env.n_agents

    import torch

    # "No-op" action: always keep the current phase (action index 0 = first green phase)
    noop_actions = torch.zeros(n_agents, dtype=torch.long)

    while True:
        td = env.step(noop_actions)

        reward = td["agents", "reward"].sum().item()
        total_reward += reward
        steps += 1

        if td["done"].item():
            break

    env.close()

    metrics = {
        "total_reward": total_reward,
        "episode_length": float(steps),
        "avg_reward_per_step": total_reward / max(steps, 1),
    }

    print("--- SUMO Baseline (Fixed-Time) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="sumo_baseline")
def main(cfg: DictConfig) -> None:
    env_cfg = maybe_to_container(cfg.env)
    run_baseline(env_cfg)


if __name__ == "__main__":
    main()
