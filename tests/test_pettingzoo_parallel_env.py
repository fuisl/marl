from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _import_parallel_env_or_skip():
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    from marl_env.pettingzoo_env import SumoTrafficParallelEnv

    return SumoTrafficParallelEnv


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

HETERO_METADATA = {
    "phase_pairs": [["N-N", "S-S"], ["S-S", "N-N"]],
    "J1": FAKE_METADATA["J1"],
    "J2": {
        "lane_sets": {
            "N-N": ["j2_l1"],
        },
        "downstream": {
            "N": None,
        },
        "fixed_timings": [10],
        "fixed_phase_order_idx": 0,
        "fixed_offset": 0,
        "pair_to_act_map": {0: 0},
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
        self._phase_durations = {0: 999, 1: 1, 2: 1, 3: 999, 4: 1, 5: 1}
        self._remaining = {tl_id: self._phase_durations[0] for tl_id in self.tl_ids}
        self._context: dict[str, dict[str, dict[int, object]]] = {}

    def start(self) -> None:
        self.time = 0
        self.phases = {tl_id: 0 for tl_id in self.tl_ids}
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
            "J1": [("veh_j1", "j1_l1", 0.0)],
            "J2": [("veh_j2", "j2_l1", 0.0)],
        }
        results: dict[str, dict[int, object]] = {}
        for veh_id, lane_id, speed in values_by_lane.get(junction_id, []):
            entry: dict[int, object] = {}
            for variable in variables:
                if variable == 81:
                    entry[variable] = lane_id
                elif variable == 86:
                    entry[variable] = 90.0
                elif variable == 114:
                    entry[variable] = 0.0
                elif variable == 64:
                    entry[variable] = speed
                elif variable == 101:
                    entry[variable] = 0.0
                elif variable == 122:
                    entry[variable] = 0.0
                elif variable == 177:
                    entry[variable] = 10.0
                elif variable == 79:
                    entry[variable] = "car"
                elif variable == 140:
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


class FakeTraCIAdapterHeteroActions(FakeTraCIAdapter):
    def get_program_logic(self, tl_id: str) -> list[FakeLogic]:
        if tl_id == "J1":
            return super().get_program_logic(tl_id)
        return [
            FakeLogic(
                [
                    FakePhase("GrGr"),
                    FakePhase("yryr"),
                    FakePhase("rrrr"),
                    FakePhase("rrrr"),
                ]
            )
        ]


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

    def build(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[10.0, 1.0], [10.0, 1.0]], dtype=torch.float32)
        return edge_index, edge_attr

    @property
    def attached_rl_ids_by_node(self) -> list[tuple[str, ...]]:
        return [(tl_id,) for tl_id in self.tl_ids]


def _patch_env_dependencies(monkeypatch: object, metadata: dict[str, object]) -> None:
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
    monkeypatch.setattr(sumo_env_mod, "get_resco_map_metadata", lambda **_: metadata)
    monkeypatch.setattr(sumo_env_mod, "tc", fake_tc)


def _make_env(monkeypatch: object, done_after: int = 20):
    from marl_env.sumo_env import TrafficSignalEnv

    SumoTrafficParallelEnv = _import_parallel_env_or_skip()
    _patch_env_dependencies(monkeypatch, FAKE_METADATA)

    core = TrafficSignalEnv(
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        delta_t=1,
        min_green_duration=0,
    )
    core.adapter = FakeTraCIAdapter(done_after=done_after)
    return SumoTrafficParallelEnv(core_env=core)


def _make_env_hetero_actions(monkeypatch: object, done_after: int = 20):
    from marl_env.sumo_env import TrafficSignalEnv

    SumoTrafficParallelEnv = _import_parallel_env_or_skip()
    _patch_env_dependencies(monkeypatch, HETERO_METADATA)

    core = TrafficSignalEnv(
        net_file="dummy.net.xml",
        route_file="dummy.rou.xml",
        delta_t=1,
        min_green_duration=0,
    )
    core.adapter = FakeTraCIAdapterHeteroActions(done_after=done_after)
    return SumoTrafficParallelEnv(core_env=core)


def test_parallel_reset_and_step_contract(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=20)

    observations, infos = env.reset()
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    assert env.agents == env.possible_agents

    actions = {
        agent: int(observations[agent]["action_mask"].argmax())
        for agent in env.agents
    }

    next_obs, rewards, terminations, truncations, next_infos = env.step(actions)

    assert set(next_obs) == set(env.possible_agents)
    assert set(rewards) == set(env.possible_agents)
    assert set(terminations) == set(env.possible_agents)
    assert set(truncations) == set(env.possible_agents)
    assert set(next_infos) == set(env.possible_agents)

    for agent in env.possible_agents:
        assert next_obs[agent]["observation"].ndim == 1
        assert next_obs[agent]["action_mask"].ndim == 1
        assert isinstance(rewards[agent], float)


def test_parallel_done_clears_agents(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=2)
    observations, _ = env.reset()

    for _ in range(10):
        actions = {
            agent: int(observations[agent]["action_mask"].argmax())
            for agent in env.agents
        }
        observations, _, terminations, truncations, _ = env.step(actions)
        if all(terminations.values()) or all(truncations.values()):
            break

    assert env.agents == []


def test_illegal_action_coerce_mode(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=10)
    observations, _ = env.reset()

    illegal_actions = {agent: 999 for agent in env.agents}
    next_obs, rewards, _, _, _ = env.step(illegal_actions)

    assert set(next_obs) == set(env.possible_agents)
    for agent in env.possible_agents:
        assert isinstance(rewards[agent], float)


def test_space_methods_available_after_reset(monkeypatch: object) -> None:
    env = _make_env(monkeypatch, done_after=10)
    env.reset()

    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        assert "observation" in obs_space.spaces
        assert "action_mask" in obs_space.spaces
        assert act_space.n >= 1


def test_per_agent_action_space_and_mask_shapes(monkeypatch: object) -> None:
    env = _make_env_hetero_actions(monkeypatch, done_after=10)
    observations, _ = env.reset()

    for agent in env.possible_agents:
        expected_n = len(env.core._green_phases[agent])
        assert env.action_space(agent).n == expected_n
        assert observations[agent]["action_mask"].shape == (expected_n,)


def test_parallel_api_compliance(monkeypatch: object) -> None:
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    env = _make_env(monkeypatch, done_after=100)
    parallel_api_test(env, num_cycles=200)
