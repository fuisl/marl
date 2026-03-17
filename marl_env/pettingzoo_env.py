"""PettingZoo Parallel environment adapter for SUMO traffic-signal control.

This wrapper preserves the current ``TrafficSignalEnv`` domain logic and only
adapts the I/O contract to PettingZoo's Parallel API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from marl_env.sumo_env import TrafficSignalEnv

try:
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency
    spaces = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None

try:
    from pettingzoo import ParallelEnv
except ImportError as exc:  # pragma: no cover - optional dependency
    ParallelEnv = None  # type: ignore[assignment]
    _PZ_IMPORT_ERROR = exc
else:
    _PZ_IMPORT_ERROR = None


_BaseParallelEnv: Any = object if ParallelEnv is None else ParallelEnv


class SumoTrafficParallelEnv(_BaseParallelEnv):
    """PettingZoo ParallelEnv wrapper over :class:`TrafficSignalEnv`.

    One PettingZoo step corresponds to one core decision step, which advances
    the simulator by ``delta_t`` SUMO seconds internally.
    """

    metadata = {
        "name": "sumo_traffic_parallel_v0",
        "render_modes": [None],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        core_env: TrafficSignalEnv | None = None,
        *,
        illegal_action_mode: str = "coerce",
        illegal_action_penalty: float = 0.0,
        **core_env_kwargs: Any,
    ) -> None:
        if _PZ_IMPORT_ERROR is not None:
            raise ImportError(
                "pettingzoo is required for SumoTrafficParallelEnv. "
                "Install with `pip install pettingzoo`."
            ) from _PZ_IMPORT_ERROR
        if _GYM_IMPORT_ERROR is not None:
            raise ImportError(
                "gymnasium is required for SumoTrafficParallelEnv. "
                "Install with `pip install gymnasium`."
            ) from _GYM_IMPORT_ERROR

        if illegal_action_mode not in {"coerce", "raise", "penalize"}:
            raise ValueError(
                "illegal_action_mode must be one of {'coerce', 'raise', 'penalize'}."
            )

        if core_env is not None and core_env_kwargs:
            raise ValueError("Pass either `core_env` or core env kwargs, not both.")

        self.core = core_env or TrafficSignalEnv(**core_env_kwargs)
        self.illegal_action_mode = illegal_action_mode
        self.illegal_action_penalty = float(illegal_action_penalty)

        self.possible_agents: list[str] = []
        self.agents: list[str] = []
        self.agent_name_mapping: dict[str, int] = {}
        self.agent_to_index: dict[str, int] = {}

        if self.core.tl_ids:
            self.possible_agents = list(self.core.tl_ids)
            self.agent_name_mapping = {
                agent: i for i, agent in enumerate(self.possible_agents)
            }
            self.agent_to_index = dict(self.agent_name_mapping)

        self._action_spaces: dict[str, Any] = {}
        self._observation_spaces: dict[str, Any] = {}
        self._num_actions_by_agent: dict[str, int] = {}
        self._last_action_mask: dict[str, np.ndarray] = {}
        self._initialized = False

        # Static graph metadata, exposed as helper accessors.
        self.edge_index: torch.Tensor | None = None
        self.edge_attr: torch.Tensor | None = None

    def observation_space(self, agent: str) -> Any:
        self._ensure_ready_for_spaces(agent)
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> Any:
        self._ensure_ready_for_spaces(agent)
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        del seed, options
        td = self.core.reset()
        self._initialize_from_core(td)
        self.agents = self.possible_agents[:]
        observations, infos = self._build_obs_infos(td)
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        missing = set(self.agents) - set(actions)
        extra = set(actions) - set(self.agents)
        if missing or extra:
            raise ValueError(
                f"Actions keys must match current agents exactly. Missing={missing}, extra={extra}."
            )

        action_tensor, penalties = self._encode_actions(actions)
        td = self.core.step(action_tensor)
        observations, infos = self._build_obs_infos(td)
        rewards = self._build_rewards(td, penalties)
        terminations, truncations = self._build_done_flags(td)

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self.core.close()

    def render(self) -> None:
        return None

    def get_static_graph_tensors(self) -> dict[str, torch.Tensor | None]:
        """Return cached static graph tensors from the core environment."""
        return {
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
        }

    def get_agent_index_map(self) -> dict[str, int]:
        """Return stable agent name to index mapping."""
        return dict(self.agent_to_index)

    def _initialize_from_core(self, td: Any) -> None:
        current_ids = list(self.core.tl_ids)
        if not self._initialized:
            self.possible_agents = current_ids
            self.agent_name_mapping = {agent: i for i, agent in enumerate(current_ids)}
            self.agent_to_index = dict(self.agent_name_mapping)

            obs_dim = int(self.core.observation_dim)
            assert spaces is not None

            self._num_actions_by_agent = {
                agent: len(self.core._green_phases[agent]) for agent in current_ids
            }
            for agent in current_ids:
                n_actions = self._num_actions_by_agent[agent]
                self._action_spaces[agent] = spaces.Discrete(n_actions)
                self._observation_spaces[agent] = spaces.Dict(
                    {
                        "observation": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(obs_dim,),
                            dtype=np.float32,
                        ),
                        "action_mask": spaces.Box(
                            low=0,
                            high=1,
                            shape=(n_actions,),
                            dtype=np.int8,
                        ),
                    }
                )

            self._initialized = True
        elif current_ids != self.possible_agents:
            raise RuntimeError(
                "Traffic-light IDs changed across resets. "
                "SumoTrafficParallelEnv expects a fixed agent set."
            )

        self.edge_index = td.get("edge_index", None)
        self.edge_attr = td.get("edge_attr", None)

    def _ensure_ready_for_spaces(self, agent: str) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Spaces are not available before first reset(). Call reset() first."
            )
        if agent not in self._action_spaces:
            raise KeyError(f"Unknown agent {agent!r}.")

    def _build_obs_infos(
        self, td: Any
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        obs_arr = td["agents", "observation"].detach().cpu().numpy().astype(np.float32)
        mask_arr = td["agents", "action_mask"].detach().cpu().numpy().astype(np.int8)

        observations: dict[str, dict[str, np.ndarray]] = {}
        infos: dict[str, dict[str, Any]] = {}
        self._last_action_mask = {}
        for i, agent in enumerate(self.agents):
            n_actions = self._num_actions_by_agent[agent]
            agent_mask = mask_arr[i, :n_actions].copy()
            self._last_action_mask[agent] = agent_mask
            observations[agent] = {
                "observation": obs_arr[i],
                "action_mask": agent_mask,
            }
            infos[agent] = {}
        return observations, infos

    def _build_rewards(
        self, td: Any, penalties: dict[str, float]
    ) -> dict[str, float]:
        reward_arr = td["agents", "reward"].detach().cpu().numpy().reshape(-1)
        rewards: dict[str, float] = {}
        for i, agent in enumerate(self.agents):
            rewards[agent] = float(reward_arr[i] + penalties.get(agent, 0.0))
        return rewards

    def _build_done_flags(self, td: Any) -> tuple[dict[str, bool], dict[str, bool]]:
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}

        if td.get(("agents", "terminated"), None) is not None:
            terminated_arr = td["agents", "terminated"].detach().cpu().numpy().reshape(-1)
            truncated_arr = td["agents", "truncated"].detach().cpu().numpy().reshape(-1)
            for i, agent in enumerate(self.agents):
                terminations[agent] = bool(terminated_arr[i])
                truncations[agent] = bool(truncated_arr[i])
            return terminations, truncations

        if td.get(("agents", "done"), None) is not None:
            done_arr = td["agents", "done"].detach().cpu().numpy().reshape(-1)
            for i, agent in enumerate(self.agents):
                terminations[agent] = bool(done_arr[i])
                truncations[agent] = False
            return terminations, truncations

        is_done = bool(td.get("done", torch.tensor([False])).item())
        for agent in self.agents:
            terminations[agent] = is_done
            truncations[agent] = False
        return terminations, truncations

    def _encode_actions(
        self, actions: dict[str, int]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self._last_action_mask:
            raise RuntimeError("No action mask available. Call reset() before step().")

        encoded: list[int] = []
        penalties: dict[str, float] = {}

        for agent in self.agents:
            action = int(actions[agent])
            mask = self._last_action_mask[agent]
            legal = 0 <= action < mask.shape[0] and bool(mask[action])

            if not legal:
                if self.illegal_action_mode == "raise":
                    raise ValueError(
                        f"Illegal action {action} for agent {agent}. mask={mask.tolist()}"
                    )

                valid_idx = np.flatnonzero(mask)
                if valid_idx.size == 0:
                    coerced = 0
                else:
                    coerced = int(valid_idx[0])

                if self.illegal_action_mode == "penalize":
                    penalties[agent] = penalties.get(agent, 0.0) + self.illegal_action_penalty

                action = coerced

            encoded.append(action)

        return torch.tensor(encoded, dtype=torch.long), penalties
