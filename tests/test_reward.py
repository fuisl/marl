from __future__ import annotations

import torch

from marl_env.reward import IntersectionMetrics, RewardCalculator


def test_queue_reward() -> None:
    calc = RewardCalculator(mode="queue")
    metrics = IntersectionMetrics(queue_lengths=[1, 2, 3])
    assert calc.compute(metrics) == -2


def test_wait_reward() -> None:
    calc = RewardCalculator(mode="wait")
    metrics = IntersectionMetrics(waiting_times=[2.0, 3.0])
    assert calc.compute(metrics) == -2.5


def test_pressure_reward_uses_pressure_field() -> None:
    calc = RewardCalculator(mode="pressure")
    metrics = IntersectionMetrics(pressure=7.0)
    assert calc.compute(metrics) == -7.0


def test_pressure_queue_reward_normalized_queue_penalty() -> None:
    calc = RewardCalculator(mode="pressure_queue", weights={"pressure": 1.0, "queue": 0.1})
    metrics = IntersectionMetrics(queue_lengths=[2.0, 4.0], pressure=3.0)
    # -|3| - 0.1 * ((2+4)/2)
    assert calc.compute(metrics) == -3.3


def test_combined_reward_finite() -> None:
    calc = RewardCalculator(mode="combined")
    metrics = IntersectionMetrics(
        queue_lengths=[1, 2],
        waiting_times=[3.0, 4.0],
        mean_speeds=[5.0, 7.0],
        occupancies=[0.2, 0.3],
        throughput=2,
    )
    reward = calc.compute(metrics)
    assert isinstance(reward, float)
    assert torch.isfinite(torch.tensor(reward))


def test_batch_reward_shape_and_finite() -> None:
    calc = RewardCalculator(mode="combined")
    rewards = calc.compute_batch(
        [
            IntersectionMetrics(queue_lengths=[1], waiting_times=[1.5], mean_speeds=[3.0]),
            IntersectionMetrics(queue_lengths=[2], waiting_times=[2.5], mean_speeds=[4.0]),
        ]
    )
    assert rewards.shape == (2,)
    assert torch.isfinite(rewards).all()
