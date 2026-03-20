from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marl_env.resco_reporting import to_public_metrics


TRAIN_LOG_FIELDNAMES = [
    "Episode",
    "Episode Length",
    "Total Transitions",
    "Elapsed Time (s)",
    "Avg Duration",
    "Avg Waiting Time",
    "Avg Time Loss",
    "Avg Queue Length",
    "Avg Reward",
    "Global Reward",
]


def build_train_log_row(
    *,
    episode: int,
    n_steps: int,
    validation_metrics: Mapping[str, float],
    total_transitions: int,
    elapsed_s: float,
) -> dict[str, Any]:
    public = to_public_metrics(dict(validation_metrics))
    return {
        "Episode": int(episode),
        "Episode Length": int(n_steps),
        "Total Transitions": int(total_transitions),
        "Elapsed Time (s)": round(float(elapsed_s), 1),
        "Avg Duration": round(public["Avg Duration"], 4),
        "Avg Waiting Time": round(public["Avg Waiting Time"], 4),
        "Avg Time Loss": round(public["Avg Time Loss"], 4),
        "Avg Queue Length": round(public["Avg Queue Length"], 4),
        "Avg Reward": round(public["Avg Reward"], 4),
        "Global Reward": round(public["Global Reward"], 4),
    }


def build_train_wandb_payload(
    *,
    episode: int,
    n_steps: int,
    validation_metrics: Mapping[str, float],
    total_transitions: int,
    elapsed_s: float,
    best_global_reward: float,
) -> dict[str, float | int]:
    public = to_public_metrics(dict(validation_metrics))
    return {
        "Episode": int(episode),
        "Episode Length": int(n_steps),
        "Total Transitions": int(total_transitions),
        "Elapsed Time (s)": float(elapsed_s),
        "Avg Duration": float(public["Avg Duration"]),
        "Avg Waiting Time": float(public["Avg Waiting Time"]),
        "Avg Time Loss": float(public["Avg Time Loss"]),
        "Avg Queue Length": float(public["Avg Queue Length"]),
        "Avg Reward": float(public["Avg Reward"]),
        "Global Reward": float(public["Global Reward"]),
        "Best Global Reward": float(best_global_reward),
    }


def print_train_progress(
    *,
    episode: int,
    validation_metrics: Mapping[str, float],
    total_transitions: int,
) -> None:
    public = to_public_metrics(dict(validation_metrics))
    print(
        f"{episode:6d}  {public['Global Reward']:12.2f}"
        f"  {public['Avg Reward']:10.2f}"
        f"  {public['Avg Duration']:10.2f}"
        f"  {public['Avg Waiting Time']:10.2f}"
        f"  {public['Avg Time Loss']:10.2f}"
        f"  {public['Avg Queue Length']:10.2f}"
        f"  {total_transitions:9d}"
    )
