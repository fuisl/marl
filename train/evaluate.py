"""Evaluation script — run a trained raw-loop agent on SUMO and report metrics.

Usage::

    python -m train.evaluate runtime.checkpoint_path=runs/gat_baseline/best_agent.pt
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from config_utils import load_dotenv, resolve_repo_path
from marl_env.resco_reporting import to_public_metrics
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
    output_dir: str | None = None,
) -> list[dict[str, float]]:
    """Load a checkpoint and evaluate for ``n_episodes``.

    Returns
    -------
    list of dicts
        Per-episode metrics (episode_return, episode_length, ...).
    """
    # Override GUI setting for visual evaluation
    env_cfg = {**env_cfg, "gui": gui}
    if str(env_cfg.get("benchmark_mode", "native")) != "resco":
        raise ValueError("The public evaluation benchmark path requires benchmark_mode='resco'.")
    if output_dir not in (None, ""):
        env_cfg.setdefault("benchmark_output_dir", output_dir)

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

    all_metrics: list[dict[str, float]] = []
    for ep in range(n_episodes):
        td = env.reset().to(device)
        steps = 0

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
            steps += 1

            if next_td["done"].item():
                break
            td = next_td

        episode_kpis = dict(env.get_episode_kpis())
        enriched_info = {
            "Episode": float(ep + 1),
            "Episode Length": float(steps),
            **to_public_metrics(episode_kpis),
        }
        print(f"Episode {ep + 1}/{n_episodes}: {enriched_info}")
        all_metrics.append(enriched_info)

    env.close()

    # Summary
    avg_length = sum(m["Episode Length"] for m in all_metrics) / n_episodes
    avg_duration = sum(m["Avg Duration"] for m in all_metrics) / n_episodes
    avg_wait = sum(m["Avg Waiting Time"] for m in all_metrics) / n_episodes
    avg_time_loss = sum(m["Avg Time Loss"] for m in all_metrics) / n_episodes
    avg_queue = sum(m["Avg Queue Length"] for m in all_metrics) / n_episodes
    avg_reward = sum(m["Avg Reward"] for m in all_metrics) / n_episodes
    global_reward = sum(m["Global Reward"] for m in all_metrics) / n_episodes
    print(f"\n--- Evaluation summary ({n_episodes} episodes) ---")
    print(f"  Episode Length:  {avg_length:.1f}")
    print(f"  Avg Duration:    {avg_duration:.2f}")
    print(f"  Avg Waiting Time:{avg_wait:.2f}")
    print(f"  Avg Time Loss:   {avg_time_loss:.2f}")
    print(f"  Avg Queue Length:{avg_queue:.2f}")
    print(f"  Avg Reward:      {avg_reward:.2f}")
    print(f"  Global Reward:   {global_reward:.2f}")

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
        output_dir=str(resolve_repo_path(cfg.runtime.get("out_dir", "."))),
    )


if __name__ == "__main__":
    main()
