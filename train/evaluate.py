"""Evaluation script — run a trained raw-loop agent on SUMO and report metrics.

Usage::

    python -m train.evaluate runtime.checkpoint_path=runs/gat_baseline/best_agent.pt
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from config_utils import load_dotenv, resolve_repo_path
from marl_env.sumo_env import TrafficSignalEnv
from models.marl_discrete_sac import MARLDiscreteSAC

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

    avg_metric_keys = (
        "avg_delay_s",
        "avg_queue_length",
        "avg_speed_mps",
        "avg_occupancy_pct",
        "min_expected_vehicles",
        "network_total_waiting_s",
        "network_total_vehicles",
    )
    total_metric_keys = (
        "arrived_vehicles",
        "departed_vehicles",
        "teleported_vehicles",
    )

    all_metrics: list[dict[str, float]] = []
    for ep in range(n_episodes):
        td = env.reset().to(device)
        total_reward = 0.0
        steps = 0
        metric_sums: dict[str, float] = {k: 0.0 for k in avg_metric_keys + total_metric_keys}

        while True:
            obs = td.get("graph_observation", td["agents", "observation"])
            edge_index = td["edge_index"]
            edge_attr = td.get("edge_attr", None)
            action_mask = td["agents", "action_mask"]

            actions, _ = agent.select_action(
                obs,
                edge_index,
                edge_attr,
                action_mask,
                deterministic=True,
                agent_node_indices=td["agent_node_indices"],
                agent_node_mask=td["agent_node_mask"],
            )

            next_td = env.step(actions.cpu()).to(device)
            total_reward += float(next_td["agents", "reward"].sum().item())
            interval = env.get_interval_kpis()
            for key in avg_metric_keys + total_metric_keys:
                metric_sums[key] += float(interval.get(key, 0.0))
            steps += 1

            if next_td["done"].item():
                break
            td = next_td

        episode_kpis = env.get_episode_kpis()
        enriched_info = {
            "episode_return": total_reward,
            "episode_length": float(steps),
            "avg_travel_time_s": float(episode_kpis.get("avg_travel_time_s", 0.0)),
            "avg_delay_s": metric_sums["avg_delay_s"] / max(steps, 1),
            "avg_queue_length": metric_sums["avg_queue_length"] / max(steps, 1),
            "avg_speed_mps": metric_sums["avg_speed_mps"] / max(steps, 1),
            "avg_occupancy_pct": metric_sums["avg_occupancy_pct"] / max(steps, 1),
            "avg_min_expected_vehicles": metric_sums["min_expected_vehicles"] / max(steps, 1),
            "avg_network_waiting_s_per_decision": metric_sums["network_total_waiting_s"] / max(steps, 1),
            "avg_network_vehicles_per_decision": metric_sums["network_total_vehicles"] / max(steps, 1),
            "total_arrived_vehicles": metric_sums["arrived_vehicles"],
            "total_departed_vehicles": metric_sums["departed_vehicles"],
            "total_teleported_vehicles": metric_sums["teleported_vehicles"],
        }
        print(f"Episode {ep + 1}/{n_episodes}: {enriched_info}")
        all_metrics.append(enriched_info)

    env.close()

    # Summary
    avg_return = sum(m["episode_return"] for m in all_metrics) / n_episodes
    avg_length = sum(m["episode_length"] for m in all_metrics) / n_episodes
    avg_travel_time = sum(m["avg_travel_time_s"] for m in all_metrics) / n_episodes
    avg_delay = sum(m["avg_delay_s"] for m in all_metrics) / n_episodes
    print(f"\n--- Evaluation summary ({n_episodes} episodes) ---")
    print(f"  Avg return:  {avg_return:.2f}")
    print(f"  Avg length:  {avg_length:.1f}")
    print(f"  Avg travel time: {avg_travel_time:.2f} s")
    print(f"  Avg delay: {avg_delay:.2f} s")

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
