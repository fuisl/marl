from __future__ import annotations

import pytest


def _import_parallel_env_or_skip():
    pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    from marl_env.pettingzoo_env import SumoTrafficParallelEnv

    return SumoTrafficParallelEnv


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
        self._phase_durations = {0: 999, 1: 1, 2: 1, 3: 999, 4: 1, 5: 1}
        self._remaining = {tl_id: self._phase_durations[0] for tl_id in self.tl_ids}

    def start(self) -> None:
        self.time = 0
        self.phases = {tl_id: 0 for tl_id in self.tl_ids}
        self._remaining = {tl_id: self._phase_durations[0] for tl_id in self.tl_ids}

    def close(self, wait: bool = False) -> None:
        _ = wait

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


class FakeTraCIAdapterHeteroActions(FakeTraCIAdapter):
    def get_program_logic(self, tl_id: str) -> list[FakeLogic]:
        if tl_id == "J1":
            # Two controllable greens: phases 0 and 3
            return super().get_program_logic(tl_id)

        # One controllable green: phase 0 only
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
    def __init__(self, net_file: str, tl_ids: list[str]) -> None:
        self.net_file = net_file
        self.tl_ids = tl_ids

    def build(self):
        import torch

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor([[10.0, 1.0], [10.0, 1.0]], dtype=torch.float32)
        return edge_index, edge_attr


def _make_env(monkeypatch: object, done_after: int = 20):
    from marl_env.sumo_env import TrafficSignalEnv
    import marl_env.sumo_env as sumo_env_mod

    SumoTrafficParallelEnv = _import_parallel_env_or_skip()
    monkeypatch.setattr(sumo_env_mod, "GraphBuilder", FakeGraphBuilder)

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
    import marl_env.sumo_env as sumo_env_mod

    SumoTrafficParallelEnv = _import_parallel_env_or_skip()
    monkeypatch.setattr(sumo_env_mod, "GraphBuilder", FakeGraphBuilder)

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
