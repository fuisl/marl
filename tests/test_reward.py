from __future__ import annotations

from types import SimpleNamespace

from marl_env.reward import available_reward_names, compute_rewards


class _FakeObservation:
    def __init__(self, *, total_wait: float, total_queued: float, lane_queues: dict[str, float]) -> None:
        self.total_wait = total_wait
        self.total_queued = total_queued
        self._lane_queues = dict(lane_queues)

    def get_lane(self, lane_id: str) -> SimpleNamespace:
        return SimpleNamespace(queued=float(self._lane_queues[lane_id]))


def test_available_reward_names_are_resco_compatible() -> None:
    assert available_reward_names() == ("pressure", "wait")


def test_wait_reward_matches_total_wait() -> None:
    signals = {
        "J1": SimpleNamespace(
            observation=_FakeObservation(total_wait=12.5, total_queued=0.0, lane_queues={}),
            outbound_lanes=[],
            out_lane_to_signal_id={},
        )
    }
    rewards = compute_rewards(reward_name="wait", signals=signals)
    assert rewards == {"J1": -12.5}


def test_pressure_reward_uses_signed_downstream_balance() -> None:
    signals = {
        "J1": SimpleNamespace(
            observation=_FakeObservation(total_wait=0.0, total_queued=7.0, lane_queues={}),
            outbound_lanes=["l_out"],
            out_lane_to_signal_id={"l_out": "J2"},
        ),
        "J2": SimpleNamespace(
            observation=_FakeObservation(total_wait=0.0, total_queued=2.0, lane_queues={"l_out": 3.0}),
            outbound_lanes=[],
            out_lane_to_signal_id={},
        ),
    }
    rewards = compute_rewards(reward_name="pressure", signals=signals)
    assert rewards["J1"] == -(7.0 - 3.0)


def test_unknown_reward_name_raises() -> None:
    signals = {
        "J1": SimpleNamespace(
            observation=_FakeObservation(total_wait=0.0, total_queued=0.0, lane_queues={}),
            outbound_lanes=[],
            out_lane_to_signal_id={},
        )
    }
    try:
        compute_rewards(reward_name="queue", signals=signals)
    except ValueError as exc:
        assert "Unknown reward" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected unknown reward selection to fail.")
