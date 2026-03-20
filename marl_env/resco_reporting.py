"""RESCO-compatible metric parsing and public benchmark summaries."""

from __future__ import annotations

import ast
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


RESCO_RAW_CSV_FIELDS = (
    "rewards",
    "max_queues",
    "queue_lengths",
    "vehicles",
    "phase_length",
)

RESCO_PRETTY_METRICS = {
    "duration": "Avg Duration",
    "waitingTime": "Avg Waiting Time",
    "timeLoss": "Avg Time Loss",
    "queue_lengths": "Avg Queue Length",
    "rewards": "Avg Reward",
}

RESCO_PUBLIC_METRIC_ORDER = (
    "Avg Duration",
    "Avg Waiting Time",
    "Avg Time Loss",
    "Avg Queue Length",
    "Avg Reward",
    "Global Reward",
)


def parse_tripinfo_metrics(tripinfo_path: str | Path) -> dict[str, float]:
    tripinfo_file = Path(tripinfo_path)
    root = ET.parse(tripinfo_file).getroot()

    totals = {
        "duration": 0.0,
        "waitingTime": 0.0,
        "timeLoss": 0.0,
    }
    num_trips = 0
    for child in root:
        veh_id = str(child.attrib.get("id", ""))
        if veh_id.startswith("ghost"):
            continue
        num_trips += 1
        totals["duration"] += float(child.attrib.get("duration", 0.0))
        totals["waitingTime"] += float(child.attrib.get("waitingTime", 0.0))
        totals["timeLoss"] += float(child.attrib.get("timeLoss", 0.0))
        totals["timeLoss"] += float(child.attrib.get("departDelay", 0.0))

    denom = max(num_trips, 1)
    return {key: value / denom for key, value in totals.items()}


def parse_metrics_csv(metrics_path: str | Path) -> dict[str, float]:
    metrics_file = Path(metrics_path)
    totals = {metric: 0.0 for metric in RESCO_RAW_CSV_FIELDS}
    num_steps = 0

    with metrics_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            num_steps += 1
            for metric in RESCO_RAW_CSV_FIELDS:
                value = row.get(metric, "{}")
                try:
                    parsed = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    parsed = {}
                if not isinstance(parsed, dict) or not parsed:
                    step_avg = 0.0
                else:
                    step_total = sum(float(v) for v in parsed.values())
                    step_avg = step_total / max(len(parsed), 1)
                totals[metric] += step_avg

    denom = max(num_steps, 1)
    return {key: value / denom for key, value in totals.items()}


def load_episode_raw_metrics(
    *,
    tripinfo_path: str | Path,
    metrics_path: str | Path,
    global_reward: float,
) -> dict[str, float]:
    tripinfo_metrics = parse_tripinfo_metrics(tripinfo_path)
    csv_metrics = parse_metrics_csv(metrics_path)
    merged = {**tripinfo_metrics, **csv_metrics}
    merged["global_reward"] = float(global_reward)
    return merged


def to_public_metrics(raw_metrics: dict[str, float]) -> dict[str, float]:
    public = {
        RESCO_PRETTY_METRICS["duration"]: float(raw_metrics.get("duration", 0.0)),
        RESCO_PRETTY_METRICS["waitingTime"]: float(raw_metrics.get("waitingTime", 0.0)),
        RESCO_PRETTY_METRICS["timeLoss"]: float(raw_metrics.get("timeLoss", 0.0)),
        RESCO_PRETTY_METRICS["queue_lengths"]: float(raw_metrics.get("queue_lengths", 0.0)),
        RESCO_PRETTY_METRICS["rewards"]: float(raw_metrics.get("rewards", 0.0)),
        "Global Reward": float(raw_metrics.get("global_reward", 0.0)),
    }
    return public
