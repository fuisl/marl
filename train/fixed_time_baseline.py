"""Run SUMO with its default fixed-time traffic signal programs (baseline).

Collects the same traffic metrics that the RL agent will be evaluated on,
so you have a fair comparison baseline.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from config_utils import load_dotenv

from marl_env.sumo_env import TrafficSignalEnv

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

load_dotenv()


def run_baseline(
    env_cfg: dict,
    *,
    out_dir: str | Path | None = None,
    wandb_cfg: dict[str, Any] | None = None,
    run_name: str | None = None,
    seed: int = 0,
) -> dict[str, float]:
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
    metric_sums: dict[str, float] = {k: 0.0 for k in avg_metric_keys + total_metric_keys}

    t0 = time.perf_counter()
    while True:
        td = env.step(noop_actions)

        reward = td["agents", "reward"].sum().item()
        total_reward += reward
        interval_kpis = env.get_interval_kpis()
        for key in avg_metric_keys + total_metric_keys:
            metric_sums[key] += float(interval_kpis.get(key, 0.0))
        steps += 1

        if td["done"].item():
            break

    episode_kpis = env.get_episode_kpis()
    env.close()

    validation_metrics = {
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
        "avg_travel_time_s": float(episode_kpis.get("avg_travel_time_s", 0.0)),
    }

    elapsed = time.perf_counter() - t0
    metrics = {
        "total_reward": total_reward,
        "episode_length": float(steps),
        "avg_reward_per_step": total_reward / max(steps, 1),
        **validation_metrics,
    }

    print("--- SUMO Baseline (Fixed-Time) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Persist baseline metrics with the same CSV schema as RL training logs.
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "train_log.csv"
        fieldnames = [
            "episode",
            "n_steps",
            "return",
            "return_ma50",
            "return_per_agent_step",
            "traffic_avg_travel_time_s",
            "traffic_avg_delay_s",
            "traffic_avg_queue_length",
            "val_avg_speed_mps",
            "val_avg_occupancy_pct",
            "val_avg_min_expected_vehicles",
            "traffic_network_total_waiting_s",
            "traffic_network_total_vehicles",
            "val_total_arrived_vehicles",
            "val_total_departed_vehicles",
            "val_total_teleported_vehicles",
            "critic_loss",
            "actor_loss",
            "alpha_loss",
            "q1",
            "q2",
            "entropy",
            "alpha",
            "total_transitions",
            "elapsed_s",
        ]
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "episode": 1,
                    "n_steps": steps,
                    "return": round(total_reward, 3),
                    "return_ma50": round(total_reward, 3),
                    "return_per_agent_step": round(
                        total_reward / max(steps * max(n_agents, 1), 1),
                        6,
                    ),
                    "traffic_avg_travel_time_s": round(validation_metrics["avg_travel_time_s"], 4),
                    "traffic_avg_delay_s": round(validation_metrics["avg_delay_s"], 4),
                    "traffic_avg_queue_length": round(validation_metrics["avg_queue_length"], 4),
                    "val_avg_speed_mps": round(validation_metrics["avg_speed_mps"], 4),
                    "val_avg_occupancy_pct": round(validation_metrics["avg_occupancy_pct"], 4),
                    "val_avg_min_expected_vehicles": round(validation_metrics["avg_min_expected_vehicles"], 4),
                    "traffic_network_total_waiting_s": round(
                        validation_metrics["avg_network_waiting_s_per_decision"], 4
                    ),
                    "traffic_network_total_vehicles": round(
                        validation_metrics["avg_network_vehicles_per_decision"], 4
                    ),
                    "val_total_arrived_vehicles": round(validation_metrics["total_arrived_vehicles"], 2),
                    "val_total_departed_vehicles": round(validation_metrics["total_departed_vehicles"], 2),
                    "val_total_teleported_vehicles": round(validation_metrics["total_teleported_vehicles"], 2),
                    "critic_loss": "",
                    "actor_loss": "",
                    "alpha_loss": "",
                    "q1": "",
                    "q2": "",
                    "entropy": "",
                    "alpha": "",
                    "total_transitions": steps,
                    "elapsed_s": round(elapsed, 1),
                }
            )
        print(f"  train_log_csv: {csv_path}")

    use_wandb = (
        _WANDB_AVAILABLE
        and isinstance(wandb_cfg, dict)
        and bool(wandb_cfg.get("enabled", False))
    )
    if use_wandb:
        wb_name = str(run_name or wandb_cfg.get("run_name") or "fixed_time_baseline")
        wandb.init(
            project=str(wandb_cfg.get("project", "marl-traffic-gat")),
            name=wb_name,
            config={
                "env": env_cfg,
                "seed": seed,
                "algo": "fixed_time_baseline",
            },
            dir=str(Path(out_dir) if out_dir is not None else Path.cwd()),
            tags=["baseline", "fixed-time"],
            settings=wandb.Settings(start_method="thread"),
        )
        wandb.define_metric("Episode")
        wandb.define_metric("Traffic/*", step_metric="Episode")
        wandb.define_metric("Learning/*", step_metric="Episode")
        wandb.define_metric("RESCO/*", step_metric="Episode")
        wandb.log(
            {
                "Episode": 1,
                "Learning/Reward": total_reward,
                "Learning/Reward (MA50)": total_reward,
                "Learning/Reward Per Agent-Step": total_reward / max(steps * max(n_agents, 1), 1),
                "Learning/Episode Length (steps)": float(steps),
                "Learning/Total Transitions": float(steps),
                "Learning/Elapsed Time (s)": elapsed,
                "Traffic/Average Travel Time (s)": validation_metrics["avg_travel_time_s"],
                "Traffic/Average Waiting Time (s)": validation_metrics["avg_delay_s"],
                "Traffic/Average Delay Proxy (s)": validation_metrics["avg_delay_s"],
                "Traffic/Average Queue Length": validation_metrics["avg_queue_length"],
                "Traffic/Average Speed (m/s)": validation_metrics["avg_speed_mps"],
                "Traffic/Average Occupancy (%)": validation_metrics["avg_occupancy_pct"],
                "Traffic/Average Min Expected Vehicles": validation_metrics["avg_min_expected_vehicles"],
                "Traffic/Average Network Waiting (veh*s per decision)": validation_metrics[
                    "avg_network_waiting_s_per_decision"
                ],
                "Traffic/Average Network Vehicles (per decision)": validation_metrics[
                    "avg_network_vehicles_per_decision"
                ],
                "Traffic/Arrived Vehicles": validation_metrics["total_arrived_vehicles"],
                "Traffic/Departed Vehicles": validation_metrics["total_departed_vehicles"],
                "Traffic/Teleported Vehicles": validation_metrics["total_teleported_vehicles"],
                "RESCO/duration": validation_metrics["avg_travel_time_s"],
                "RESCO/waitingTime": validation_metrics["avg_delay_s"],
                "RESCO/timeLoss_proxy": validation_metrics["avg_delay_s"],
                "RESCO/queue_lengths": validation_metrics["avg_queue_length"],
                "RESCO/rewards": total_reward,
                "RESCO/vehicles": validation_metrics["total_departed_vehicles"],
            }
        )
        wandb.run.summary["run_name"] = wb_name
        wandb.run.summary["algo"] = "fixed_time_baseline"
        wandb.finish()

    return metrics

