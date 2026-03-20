"""Unified reward registry for the RESCO-native environment."""

from __future__ import annotations

from collections.abc import Callable

from marl_env.resco_observation import RescoSignalState


RewardFn = Callable[[dict[str, RescoSignalState]], dict[str, float]]


def _wait_reward(signals: dict[str, RescoSignalState]) -> dict[str, float]:
    return {
        signal_id: -float(signal.observation.total_wait)
        for signal_id, signal in signals.items()
    }


def _pressure_reward(signals: dict[str, RescoSignalState]) -> dict[str, float]:
    rewards: dict[str, float] = {}
    for signal_id, signal in signals.items():
        entering_queued = float(signal.observation.total_queued)
        exiting_queued = 0.0
        for lane_id in signal.outbound_lanes:
            downstream_signal_id = signal.out_lane_to_signal_id.get(lane_id)
            if downstream_signal_id in signals:
                exiting_queued += float(
                    signals[downstream_signal_id].observation.get_lane(lane_id).queued
                )
        rewards[signal_id] = -(entering_queued - exiting_queued)
    return rewards


REWARD_REGISTRY: dict[str, RewardFn] = {
    "wait": _wait_reward,
    "pressure": _pressure_reward,
}


def available_reward_names() -> tuple[str, ...]:
    return tuple(sorted(REWARD_REGISTRY))


def compute_rewards(
    *,
    reward_name: str,
    signals: dict[str, RescoSignalState],
) -> dict[str, float]:
    if reward_name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward {reward_name!r}. Available rewards: {available_reward_names()}."
        )
    return REWARD_REGISTRY[reward_name](signals)
