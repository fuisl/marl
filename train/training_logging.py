from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TRAIN_LOG_FIELDNAMES = [
    "episode", "n_steps", "return", "return_ma50",
    "return_per_agent_step",
    "traffic_avg_travel_time_s", "traffic_avg_wait_s",
    "traffic_avg_delay_s", "traffic_avg_queue_length",
    "val_avg_speed_mps", "val_avg_occupancy_pct",
    "val_avg_min_expected_vehicles",
    "traffic_network_total_waiting_s", "traffic_network_total_vehicles",
    "val_total_arrived_vehicles", "val_total_departed_vehicles",
    "val_total_teleported_vehicles",
    "critic_loss", "actor_loss", "alpha_loss",
    "q1", "q2", "entropy", "alpha",
    "total_transitions", "elapsed_s",
]


def build_train_log_row(
    *,
    episode: int,
    n_steps: int,
    episode_return: float,
    ma50: float,
    return_per_agent_step: float,
    validation_metrics: Mapping[str, float],
    total_transitions: int,
    elapsed_s: float,
    last_metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "episode": episode,
        "n_steps": n_steps,
        "return": round(episode_return, 3),
        "return_ma50": round(ma50, 3),
        "return_per_agent_step": round(return_per_agent_step, 6),
        "traffic_avg_travel_time_s": round(validation_metrics["avg_travel_time_s"], 4),
        "traffic_avg_wait_s": round(validation_metrics["avg_wait_s"], 4),
        "traffic_avg_delay_s": round(validation_metrics["avg_delay_s"], 4),
        "traffic_avg_queue_length": round(validation_metrics["avg_queue_length"], 4),
        "val_avg_speed_mps": round(validation_metrics["avg_speed_mps"], 4),
        "val_avg_occupancy_pct": round(validation_metrics["avg_occupancy_pct"], 4),
        "val_avg_min_expected_vehicles": round(validation_metrics["avg_min_expected_vehicles"], 4),
        "traffic_network_total_waiting_s": round(validation_metrics["avg_network_waiting_s_per_decision"], 4),
        "traffic_network_total_vehicles": round(validation_metrics["avg_network_vehicles_per_decision"], 4),
        "val_total_arrived_vehicles": round(validation_metrics["total_arrived_vehicles"], 2),
        "val_total_departed_vehicles": round(validation_metrics["total_departed_vehicles"], 2),
        "val_total_teleported_vehicles": round(validation_metrics["total_teleported_vehicles"], 2),
        "total_transitions": total_transitions,
        "elapsed_s": round(elapsed_s, 1),
        **{k: round(float(v), 5) for k, v in last_metrics.items()},
    }


def build_train_wandb_payload(
    *,
    episode: int,
    episode_return: float,
    ma50: float,
    return_per_agent_step: float,
    n_steps: int,
    total_transitions: int,
    elapsed_s: float,
    best_return: float,
    validation_metrics: Mapping[str, float],
    last_metrics: Mapping[str, float],
) -> dict[str, float | int]:
    return {
        "Episode": episode,
        "Learning/Reward": episode_return,
        "Learning/Reward (MA50)": ma50,
        "Learning/Reward Per Agent-Step": return_per_agent_step,
        "Learning/Episode Length (steps)": n_steps,
        "Learning/Total Transitions": total_transitions,
        "Learning/Elapsed Time (s)": elapsed_s,
        "Learning/Critic Loss": last_metrics.get("critic_loss", float("nan")),
        "Learning/Actor Loss": last_metrics.get("actor_loss", float("nan")),
        "Learning/Alpha Loss": last_metrics.get("alpha_loss", float("nan")),
        "Learning/Q1 Mean": last_metrics.get("q1", float("nan")),
        "Learning/Q2 Mean": last_metrics.get("q2", float("nan")),
        "Learning/Policy Entropy": last_metrics.get("entropy", float("nan")),
        "Learning/Alpha": last_metrics.get("alpha", float("nan")),
        "Learning/Best Reward": best_return,
        "Traffic/Average Travel Time (s)": validation_metrics["avg_travel_time_s"],
        "Traffic/Average Waiting Time (s)": validation_metrics["avg_wait_s"],
        "Traffic/Average Time Loss (s)": validation_metrics["avg_delay_s"],
        "Traffic/Average Queue Length": validation_metrics["avg_queue_length"],
        "Traffic/Average Speed (m/s)": validation_metrics["avg_speed_mps"],
        "Traffic/Average Occupancy (%)": validation_metrics["avg_occupancy_pct"],
        "Traffic/Average Min Expected Vehicles": validation_metrics["avg_min_expected_vehicles"],
        "Traffic/Average Network Waiting (veh*s per decision)": validation_metrics["avg_network_waiting_s_per_decision"],
        "Traffic/Average Network Vehicles (per decision)": validation_metrics["avg_network_vehicles_per_decision"],
        "Traffic/Arrived Vehicles": validation_metrics["total_arrived_vehicles"],
        "Traffic/Departed Vehicles": validation_metrics["total_departed_vehicles"],
        "Traffic/Teleported Vehicles": validation_metrics["total_teleported_vehicles"],
        "RESCO/duration": validation_metrics["avg_travel_time_s"],
        "RESCO/waitingTime": validation_metrics["avg_wait_s"],
        "RESCO/timeLoss": validation_metrics["avg_delay_s"],
        "RESCO/queue_lengths": validation_metrics["avg_queue_length"],
        "RESCO/rewards": episode_return,
        "RESCO/vehicles": validation_metrics["total_departed_vehicles"],
    }


def print_train_progress(
    *,
    episode: int,
    episode_return: float,
    ma50: float,
    total_transitions: int,
    warmup: int,
    last_metrics: Mapping[str, float],
) -> None:
    cl = last_metrics.get("critic_loss", float("nan"))
    al = last_metrics.get("actor_loss", float("nan"))
    en = last_metrics.get("entropy", float("nan"))
    alpha = last_metrics.get("alpha", float("nan"))
    warm_tag = "" if total_transitions >= warmup else " [warmup]"
    print(
        f"{episode:6d}  {episode_return:10.2f}  {ma50:10.2f}"
        f"  {cl:9.4f}  {al:9.4f}  {en:9.4f}  {alpha:7.4f}"
        f"  {total_transitions:7d}{warm_tag}"
    )
