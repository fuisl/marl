"""Reward functions for traffic signal control.

Each function receives per-intersection metrics collected from SUMO and
returns a scalar reward for that agent.  Combine as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class IntersectionMetrics:
    """Raw metrics for a single intersection collected from SUMO."""

    queue_lengths: list[float] = field(default_factory=list)
    waiting_times: list[float] = field(default_factory=list)
    mean_speeds: list[float] = field(default_factory=list)
    occupancies: list[float] = field(default_factory=list)
    pressure: float = 0.0
    throughput: int = 0  # vehicles that cleared the intersection this step


class RewardCalculator:
    """Pluggable reward calculator.

    Parameters
    ----------
    mode : str
        One of ``"queue"``, ``"wait"``, ``"pressure"``, ``"pressure_queue"``, ``"combined"``.
    weights : dict[str, float] | None
        Weights for ``"combined"`` and ``"pressure_queue"`` modes.
    """

    MODES = ("queue", "wait", "pressure", "pressure_queue", "combined")

    def __init__(
        self,
        mode: str = "combined",
        weights: dict[str, float] | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown reward mode {mode!r}; choose from {self.MODES}")
        self.mode = mode
        self.weights = weights or {
            "queue": -0.25,
            "wait": -0.25,
            "speed": 0.25,
            "throughput": 0.25,
            "pressure": 1.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, metrics: IntersectionMetrics) -> float:
        if self.mode == "queue":
            return self._queue_reward(metrics)
        if self.mode == "wait":
            return self._wait_reward(metrics)
        if self.mode == "pressure":
            return self._pressure_reward(metrics)
        if self.mode == "pressure_queue":
            return self._pressure_queue_reward(metrics)
        return self._combined_reward(metrics)

    def compute_batch(
        self, metrics_list: list[IntersectionMetrics]
    ) -> Tensor:
        """Return reward tensor of shape ``[n_agents]``."""
        return torch.tensor(
            [self.compute(m) for m in metrics_list], dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------
    @staticmethod
    def _lane_count(m: IntersectionMetrics) -> int:
        return max(len(m.queue_lengths), len(m.waiting_times), 1)

    @staticmethod
    def _queue_reward(m: IntersectionMetrics) -> float:
        lane_count = RewardCalculator._lane_count(m)
        return -sum(m.queue_lengths) / lane_count

    @staticmethod
    def _wait_reward(m: IntersectionMetrics) -> float:
        lane_count = RewardCalculator._lane_count(m)
        return -sum(m.waiting_times) / lane_count

    @staticmethod
    def _pressure_reward(m: IntersectionMetrics) -> float:
        # Pressure penalty is the negative absolute queue imbalance.
        return -abs(float(m.pressure))

    def _pressure_queue_reward(self, m: IntersectionMetrics) -> float:
        lane_count = self._lane_count(m)
        queue_norm = sum(m.queue_lengths) / lane_count
        lambda_queue = float(self.weights.get("queue", 0.1))
        pressure_weight = float(self.weights.get("pressure", 1.0))
        return -pressure_weight * abs(float(m.pressure)) - lambda_queue * queue_norm

    def _combined_reward(self, m: IntersectionMetrics) -> float:
        w = self.weights
        lane_count = self._lane_count(m)
        queue_term = w.get("queue", 0.0) * (sum(m.queue_lengths) / lane_count)
        wait_term = w.get("wait", 0.0) * (sum(m.waiting_times) / lane_count)
        speed_term = (
            w.get("speed", 0.0) * (sum(m.mean_speeds) / max(len(m.mean_speeds), 1))
        )
        throughput_term = w.get("throughput", 0.0) * m.throughput
        return queue_term + wait_term + speed_term + throughput_term
