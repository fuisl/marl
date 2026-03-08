"""Run SUMO with its default fixed-time traffic signal programs (baseline).

Collects the same traffic metrics that the RL agent will be evaluated on,
so you have a fair comparison baseline.

Usage::

    python scripts/run_sumo_baseline.py --config configs/env.yaml
"""

from __future__ import annotations

import argparse

import yaml

from env.sumo_env import TrafficSignalEnv


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SUMO fixed-time baseline.")
    parser.add_argument("--config", type=str, default="configs/env.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        env_cfg = yaml.safe_load(f)

    run_baseline(env_cfg)


if __name__ == "__main__":
    main()
