from __future__ import annotations

from types import SimpleNamespace

import torch
from tensordict import TensorDict

from marl_env.sumo_env import TrafficSignalEnv


FAKE_METADATA = {
    "phase_pairs": [["N-N", "S-S"], ["S-S", "N-N"]],
    "J1": {
        "lane_sets": {
            "N-N": ["j1_l1"],
            "S-S": ["j1_l2"],
        },
        "downstream": {
            "N": "J2",
            "S": None,
        },
        "fixed_timings": [10, 10],
        "fixed_phase_order_idx": 0,
        "fixed_offset": 0,
        "pair_to_act_map": {0: 0, 1: 1},
    },
    "J2": {
        "lane_sets": {
            "N-N": ["j2_l1"],
            "S-S": ["j2_l2"],
        },
        "downstream": {
            "N": None,
            "S": "J1",
        },
        "fixed_timings": [10, 10],
        "fixed_phase_order_idx": 0,
        "fixed_offset": 0,
        "pair_to_act_map": {0: 0, 1: 1},
    },
}


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
        self.tripinfo_output: str | None = None
        self.tl_ids = ["J1", "J2"]
        self.lanes = {
            "J1": ["j1_l1", "j1_l2"],
            "J2": ["j2_l1", "j2_l2"],
        }
        self.phases = {tl_id: 0 for tl_id in self.tl_ids}
        self.phase_history = {tl_id: [0] for tl_id in self.tl_ids}
        self._phase_durations = {0: 999, 1: 1, 2: 1, 3: 999, 4: 1, 5: 1}
        self._remaining = {tl_id: self._phase_durations[0] for tl_id in self.tl_ids}
        self._context: dict[str, dict[str, dict[int, object]]] = {}

    def start(self) -> None:
        self.time = 0
        self.phases = {tl_id: 0 for tl_id in self.tl_ids}
        self.phase_history = {tl_id: [0] for tl_id in self.tl_ids}
        self._remaining = {tl_id: self._phase_durations[0] for tl_id in self.tl_ids}

    def close(self, wait: bool = False) -> None:
        _ = wait

    def set_tripinfo_output(self, path: str | None) -> None:
        self.tripinfo_output = path

    def simulation_step(self) -> None:
        self.time += 1
        for tl_id in self.tl_ids:
            self._remaining[tl_id] -= 1
            if self._remaining[tl_id] > 0:
                continue

            nxt = (self.phases[tl_id] + 1) % 6
            self.phases[tl_id] = nxt
            self.phase_history[tl_id].append(nxt)
            self._remaining[tl_id] = self._phase_durations[nxt]

    @property
    def min_expected_vehicles(self) -> int:
        return 1 if self.time < self.done_after else 0

    @property
    def current_time(self) -> int:
        return self.time

    @property
    def end_time(self) -> int:
        return 999999

    def get_traffic_light_ids(self) -> list[str]:
        return self.tl_ids

    def get_phase(self, tl_id: str) -> int:
        return self.phases[tl_id]

    def set_phase(self, tl_id: str, phase_index: int) -> None:
        self.phases[tl_id] = phase_index
        self.phase_history[tl_id].append(phase_index)
        self._remaining[tl_id] = self._phase_durations[phase_index]

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

    def get_arrived_number(self) -> int:
        return 0

    def get_departed_number(self) -> int:
        return 0

    def get_teleported_number(self) -> int:
        return 0

    def subscribe_junction_context(
        self,
        junction_id: str,
        radius: float,
        variables: list[int],
    ) -> None:
        _ = radius
        self._context[junction_id] = self._build_context_results(junction_id, variables)

    def get_junction_context_subscription_results(self, junction_id: str) -> dict[str, dict[int, object]]:
        return dict(self._context.get(junction_id, {}))

    def _build_context_results(
        self,
        junction_id: str,
        variables: list[int],
    ) -> dict[str, dict[int, object]]:
        values_by_lane = {
            "J1": [("veh_j1_a", "j1_l1", 0.0), ("veh_j1_b", "j1_l2", 5.0)],
            "J2": [("veh_j2_a", "j2_l1", 0.0), ("veh_j2_b", "j2_l2", 5.0)],
        }
        results: dict[str, dict[int, object]] = {}
        for veh_id, lane_id, speed in values_by_lane[junction_id]:
            entry: dict[int, object] = {}
            for variable in variables:
                if variable == 81:  # lane id placeholder when patched tc is missing
                    entry[variable] = lane_id
                elif variable == 86:  # lane position
                    entry[variable] = 90.0
                elif variable == 114:  # acceleration
                    entry[variable] = 0.0
                elif variable == 64:  # speed
                    entry[variable] = speed
                elif variable == 101:  # fuel
                    entry[variable] = 0.0
                elif variable == 122:  # waiting
                    entry[variable] = 0.0
                elif variable == 177:  # allowed speed
                    entry[variable] = 10.0
                elif variable == 79:  # type
                    entry[variable] = "car"
                elif variable == 140:  # time loss
                    entry[variable] = 0.0
            results[veh_id] = entry
        return results

    @staticmethod
    def get_lane_length(lane_id: str) -> float:
        _ = lane_id
        return 100.0

    @staticmethod
    def get_lane_max_speed(lane_id: str) -> float:
        _ = lane_id
        return 10.0


class FakeGraphBuilder:
    def __init__(
        self,
        net_file: str,
        tl_ids: list[str],
        *,
        mode: str = "all_intersections",
    ) -> None:
        self.net_file = net_file
        self.tl_ids = tl_ids
        self.mode = mode
        self.node_ids = list(tl_ids)
        self.agent_node_indices = torch.tensor([[0], [1]], dtype=torch.long)
        self.agent_node_mask = torch.tensor([[True], [True]], dtype=torch.bool)
        self.node_positions = torch.zeros((2, 2), dtype=torch.float32)
        self.net = SimpleNamespace()

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[10.0, 1.0], [10.0, 1.0]], dtype=torch.float32)
        return edge_index, edge_attr

    @property
    def attached_rl_ids_by_node(self) -> list[tuple[str, ...]]:
        return [(tl_id,) for tl_id in self.tl_ids]


def _patch_env_dependencies(monkeypatch: object) -> None:
    import marl_env.sumo_env as sumo_env_mod

    fake_tc = SimpleNamespace(
        VAR_LANE_ID=81,
        VAR_LANEPOSITION=86,
        VAR_ACCELERATION=114,
        VAR_SPEED=64,
        VAR_FUELCONSUMPTION=101,
        VAR_WAITING_TIME=122,
        VAR_ALLOWED_SPEED=177,
        VAR_TYPE=79,
        VAR_TIMELOSS=140,
    )
    monkeypatch.setattr(sumo_env_mod, "GraphBuilder", FakeGraphBuilder)
    monkeypatch.setattr(sumo_env_mod, "get_resco_map_metadata", lambda **_: FAKE_METADATA)
    monkeypatch.setattr(sumo_env_mod, "tc", fake_tc)


def _sample_valid_actions(mask: torch.Tensor) -> torch.Tensor:
    actions: list[int] = []
    for row in mask:
        valid = torch.where(row)[0]
        assert valid.numel() > 0
        actions.append(int(valid[0].item()))
    return torch.tensor(actions, dtype=torch.long)


def _make_env(monkeypatch: object, done_after: int = 20) -> TrafficSignalEnv:
    _patch_env_dependencies(monkeypatch)
    env = TrafficSignalEnv(
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        delta_t=1,
        min_green_duration=0,
    )
    env.adapter = FakeTraCIAdapter(done_after=done_after)
    return env


def test_env_reset_and_step_shapes_and_finite(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=20)
    td = env.reset()
    assert isinstance(td, TensorDict)

    obs = td["agents", "observation"]
    mask = td["agents", "action_mask"]
    graph_metadata = env.get_graph_metadata()

    assert obs.ndim == 2
    assert mask.ndim == 2
    assert obs.shape[0] == env.n_agents
    assert mask.shape == (env.n_agents, env.num_actions)
    assert obs.shape[1] == env.observation_dim
    assert graph_metadata.edge_index.shape == (2, 2)
    assert graph_metadata.agent_node_indices.shape == (env.n_agents, 1)
    assert graph_metadata.agent_node_mask.shape == (env.n_agents, 1)
    assert torch.all(mask.any(dim=1))

    obs_shape = obs.shape
    mask_shape = mask.shape

    actions = _sample_valid_actions(mask)
    td2 = env.step(actions)

    assert td2["agents", "observation"].shape == obs_shape
    assert td2["agents", "action_mask"].shape == mask_shape
    assert torch.isfinite(td2["agents", "observation"]).all()
    assert torch.isfinite(td2["agents", "reward"]).all()
    assert torch.all(td2["agents", "action_mask"].any(dim=1))


def test_env_done_eventually_true_and_metrics_are_resco_style(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=4)
    td = env.reset()

    done = bool(td.get("done", torch.tensor([False])).item())
    for _ in range(10):
        mask = td["agents", "action_mask"]
        actions = _sample_valid_actions(mask)
        td = env.step(actions)
        done = bool(td["done"].item())
        if done:
            break

    assert done
    metrics = env.get_episode_metrics()
    assert "Avg Duration" not in metrics
    assert "duration" in metrics
    assert "waitingTime" in metrics
    assert "timeLoss" in metrics
    assert "global_reward" in metrics


def test_transition_sequence_and_elapsed_reset(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=50)
    env.reset()

    assert env.num_actions >= 2
    assert env._current_green["J1"] == 0

    switch_actions = torch.tensor([1, 1], dtype=torch.long)
    elapsed_trace: list[float] = []
    for _ in range(4):
        env.step(switch_actions)
        elapsed_trace.append(env._elapsed_green["J1"])

    history = env.adapter.phase_history["J1"]
    assert 1 in history
    assert 2 in history
    assert history[-1] == 3
    assert env._current_green["J1"] == 3
    assert 0.0 in elapsed_trace
    assert not env.constraints.in_transition("J1")
