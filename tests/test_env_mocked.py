from __future__ import annotations

import torch
from tensordict import TensorDict

from env.sumo_env import TrafficSignalEnv


class FakePhase:
    def __init__(self, state: str) -> None:
        self.state = state


class FakeLogic:
    def __init__(self, phases: list[FakePhase]) -> None:
        self.phases = phases


class FakeTraCIAdapter:
    def __init__(self, done_after: int = 20) -> None:
        self.done_after = done_after
        self.time = 0
        self.tl_ids = ["J1", "J2"]
        self.lanes = {
            "J1": ["j1_l1", "j1_l2"],
            "J2": ["j2_l1", "j2_l2"],
        }
        self.phases = {tl_id: 0 for tl_id in self.tl_ids}
        self.phase_history = {tl_id: [0] for tl_id in self.tl_ids}

    def start(self) -> None:
        self.time = 0
        self.phases = {tl_id: 0 for tl_id in self.tl_ids}
        self.phase_history = {tl_id: [0] for tl_id in self.tl_ids}

    def close(self, wait: bool = False) -> None:
        _ = wait

    def simulation_step(self) -> None:
        self.time += 1

    @property
    def min_expected_vehicles(self) -> int:
        return 1 if self.time < self.done_after else 0

    @property
    def current_time(self):
        return self.time

    @property
    def end_time(self):
        return 999999  # or a configured horizon for the fake

    def get_traffic_light_ids(self) -> list[str]:
        return self.tl_ids

    def get_phase(self, tl_id: str) -> int:
        return self.phases[tl_id]

    def set_phase(self, tl_id: str, phase_index: int) -> None:
        self.phases[tl_id] = phase_index
        self.phase_history[tl_id].append(phase_index)

    def get_program_logic(self, tl_id: str) -> list[FakeLogic]:
        _ = tl_id
        return [
            FakeLogic(
                [
                    FakePhase("GrGr"),
                    FakePhase("yryr"),
                    FakePhase("rrrr"),
                    FakePhase("rGrG"),
                    FakePhase("ryry"),
                    FakePhase("rrrr"),
                ]
            )
        ]

    def get_controlled_lanes(self, tl_id: str) -> list[str]:
        return self.lanes[tl_id]

    @staticmethod
    def get_lane_halting_number(lane_id: str) -> int:
        return 1 if lane_id.endswith("1") else 2

    @staticmethod
    def get_lane_waiting_time(lane_id: str) -> float:
        return 2.0 if lane_id.endswith("1") else 3.0

    @staticmethod
    def get_lane_occupancy(lane_id: str) -> float:
        return 0.1 if lane_id.endswith("1") else 0.2

    @staticmethod
    def get_lane_mean_speed(lane_id: str) -> float:
        return 5.0 if lane_id.endswith("1") else 6.0


class FakeGraphBuilder:
    def __init__(self, net_file: str, tl_ids: list[str]) -> None:
        self.net_file = net_file
        self.tl_ids = tl_ids

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[10.0, 1.0], [10.0, 1.0]], dtype=torch.float32)
        return edge_index, edge_attr


def _sample_valid_actions(mask: torch.Tensor) -> torch.Tensor:
    actions: list[int] = []
    for row in mask:
        valid = torch.where(row)[0]
        assert valid.numel() > 0
        actions.append(int(valid[0].item()))
    return torch.tensor(actions, dtype=torch.long)


def _make_env(monkeypatch: object, done_after: int = 20) -> TrafficSignalEnv:
    import env.sumo_env as sumo_env_mod

    monkeypatch.setattr(sumo_env_mod, "GraphBuilder", FakeGraphBuilder)
    env = TrafficSignalEnv(
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        delta_t=1,
        min_green_duration=0,
    )
    env.adapter = FakeTraCIAdapter(done_after=done_after)
    env.graph_builder = None
    return env


def test_env_reset_and_step_shapes_and_finite(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=20)
    td = env.reset()
    assert isinstance(td, TensorDict)

    obs = td["agents", "observation"]
    mask = td["agents", "action_mask"]
    edge_index = td["edge_index"]

    assert obs.ndim == 2
    assert mask.ndim == 2
    assert obs.shape[0] == env.n_agents
    assert mask.shape[0] == env.n_agents
    assert edge_index.shape[0] == 2
    assert torch.all(mask.any(dim=1))

    obs_shape = obs.shape
    mask_shape = mask.shape
    edge_index_snapshot = edge_index.clone()

    actions = _sample_valid_actions(mask)
    td2 = env.step(actions)

    assert td2["agents", "observation"].shape == obs_shape
    assert td2["agents", "action_mask"].shape == mask_shape
    assert torch.equal(td2["edge_index"], edge_index_snapshot)
    assert torch.isfinite(td2["agents", "observation"]).all()
    assert torch.isfinite(td2["agents", "reward"]).all()
    assert torch.all(td2["agents", "action_mask"].any(dim=1))


def test_env_done_eventually_true(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=6)
    td = env.reset()

    done = bool(td.get("done", torch.tensor([False])).item())
    for _ in range(20):
        mask = td["agents", "action_mask"]
        assert torch.all(mask.any(dim=1))
        actions = _sample_valid_actions(mask)
        td = env.step(actions)
        assert torch.isfinite(td["agents", "reward"]).all()
        done = bool(td["done"].item())
        if done:
            break

    assert done


def test_transition_sequence_and_elapsed_reset(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=50)
    td = env.reset()
    _ = td

    assert env.num_actions >= 2
    assert env._current_green["J1"] == 0

    switch_actions = torch.tensor([1, 1], dtype=torch.long)
    for _ in range(4):
        env.step(switch_actions)

    history = env.adapter.phase_history["J1"]
    assert 1 in history  # yellow phase
    assert 2 in history  # all-red phase
    assert history[-1] == 3  # destination green

    assert env._current_green["J1"] == 3
    assert env._elapsed_green["J1"] == 0.0
    assert not env.constraints.in_transition("J1")
