"""Custom SUMO multi-agent traffic-signal environment.

Outputs TorchRL-style ``TensorDict`` with shared graph tensors at the top
level and per-agent tensors nested under ``"agents"``.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

from env.action_constraints import ActionConstraints
from env.graph_builder import GraphBuilder
from env.reward import IntersectionMetrics, RewardCalculator
from env.traci_adapter import TraCIAdapter


class TrafficSignalEnv:
    """Multi-agent traffic-signal control environment over SUMO.

    One decision every ``delta_t`` simulation seconds.  All traffic-light
    agents act synchronously (same decision frequency).

    Parameters
    ----------
    net_file : str
        Path to SUMO ``.net.xml``.
    route_file : str
        Path to SUMO ``.rou.xml`` (or ``.trips.xml``).
    delta_t : int
        Decision interval in simulation seconds.
    reward_mode : str
        Reward function mode — forwarded to :class:`RewardCalculator`.
    yellow_duration, all_red_duration, min_green_duration : int
        Phase-transition constraints.
    sumo_binary : str
        ``"sumo"`` or ``"sumo-gui"``.
    gui : bool
        Launch SUMO-GUI.
    begin_time, end_time : int
        Simulation window.
    additional_files : list[str] | None
        Extra SUMO input files (detectors etc.).
    """

    def __init__(
        self,
        net_file: str,
        route_file: str,
        *,
        delta_t: int = 5,
        reward_mode: str = "combined",
        reward_weights: dict[str, float] | None = None,
        yellow_duration: int = 3,
        all_red_duration: int = 1,
        min_green_duration: int = 5,
        sumo_binary: str = "sumo",
        gui: bool = False,
        begin_time: int = 0,
        end_time: int = 3600,
        additional_files: list[str] | None = None,
    ) -> None:
        self.delta_t = delta_t
        self.net_file = net_file

        # --- SUMO adapter ---
        self.adapter = TraCIAdapter(
            net_file=net_file,
            route_file=route_file,
            sumo_binary=sumo_binary,
            gui=gui,
            delta_t=delta_t,
            begin_time=begin_time,
            end_time=end_time,
            additional_files=additional_files,
        )

        # --- Reward ---
        self.reward_calc = RewardCalculator(mode=reward_mode, weights=reward_weights)

        # --- Action constraints ---
        self.constraints = ActionConstraints(
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
            min_green_duration=min_green_duration,
        )

        # Populated on reset()
        self.tl_ids: list[str] = []
        self.n_agents: int = 0
        self.graph_builder: GraphBuilder | None = None
        self.edge_index: Tensor | None = None
        self.edge_attr: Tensor | None = None

        # Per-agent caches
        self._green_phases: dict[str, list[int]] = {}
        self._controlled_lanes: dict[str, list[str]] = {}
        self._elapsed_green: dict[str, float] = {}
        self._current_green: dict[str, int] = {}

    # ==================================================================
    # Core gym-like interface
    # ==================================================================
    def reset(self) -> TensorDict:
        """Start a new episode. Returns initial observation ``TensorDict``."""
        self.adapter.close()  # no-op on first call
        self.adapter.start()

        # Discover agents
        self.tl_ids = self.adapter.get_traffic_light_ids()
        self.n_agents = len(self.tl_ids)

        # Build static graph once
        self.graph_builder = GraphBuilder(self.net_file, self.tl_ids)
        self.edge_index, self.edge_attr = self.graph_builder.build()

        # Cache phase info & register constraints per agent
        for tl_id in self.tl_ids:
            green_phases = self._extract_green_phases(tl_id)
            self._green_phases[tl_id] = green_phases
            self._controlled_lanes[tl_id] = self.adapter.get_controlled_lanes(tl_id)
            self._elapsed_green[tl_id] = 0.0
            self._current_green[tl_id] = self.adapter.get_phase(tl_id)

            self.constraints.register_agent(
                tl_id,
                num_phases=self._get_num_phases(tl_id),
                green_phase_indices=green_phases,
            )

        return self._build_tensordict()

    def step(self, actions: Tensor) -> TensorDict:
        """Execute one decision step (``delta_t`` simulation seconds).

        Parameters
        ----------
        actions : Tensor
            Integer actions of shape ``[n_agents]``.  Each value indexes
            into the agent's list of green phases.

        Returns
        -------
        TensorDict
            Next observation, reward, done, and action masks.
        """
        # --- Apply actions (with yellow transition logic) ---
        for i, tl_id in enumerate(self.tl_ids):
            desired_green = self._green_phases[tl_id][int(actions[i].item())]
            current_green = self._current_green[tl_id]

            if self.constraints.needs_transition(tl_id, current_green, desired_green):
                self.constraints.start_transition(tl_id, desired_green)
                # Will set yellow in _apply_transition below
            else:
                self._elapsed_green[tl_id] += self.delta_t

        # --- Advance simulation by delta_t seconds ---
        for _ in range(self.delta_t):
            self._apply_transitions()
            self.adapter.simulation_step()

        # --- Update current green phases after transitions ---
        for tl_id in self.tl_ids:
            dest = self.constraints.get_transition_phase_idx(tl_id)
            if dest is None:
                self._current_green[tl_id] = self.adapter.get_phase(tl_id)
                # Don't reset elapsed — it was accumulated above
            else:
                self._current_green[tl_id] = dest
                self._elapsed_green[tl_id] = 0.0

        # --- Build output ---
        td = self._build_tensordict()

        # Rewards
        rewards = self._compute_rewards()
        td["agents", "reward"] = rewards.unsqueeze(-1)  # [n_agents, 1]

        # Done
        done = self.adapter.min_expected_vehicles == 0
        td["done"] = torch.tensor([done], dtype=torch.bool)
        td["agents", "done"] = torch.full(
            (self.n_agents, 1), done, dtype=torch.bool
        )

        return td

    def close(self) -> None:
        self.adapter.close()

    # ==================================================================
    # Observation helpers
    # ==================================================================
    @property
    def observation_dim(self) -> int:
        """Dimension of per-agent observation vector."""
        # queue(n_lanes) + wait(n_lanes) + occupancy(n_lanes) +
        # speed(n_lanes) + phase_onehot(n_green) + elapsed(1)
        # The actual dim depends on max_lanes and max_green_phases.
        # This is a placeholder; real value set after first reset.
        return self._obs_dim

    @property
    def num_actions(self) -> int:
        """Max number of green-phase actions across agents."""
        if not self._green_phases:
            return 0
        return max(len(v) for v in self._green_phases.values())

    def _build_tensordict(self) -> TensorDict:
        obs_list: list[Tensor] = []

        for tl_id in self.tl_ids:
            obs_list.append(self._get_observation(tl_id))

        obs = torch.stack(obs_list, dim=0)  # [n_agents, d_obs]
        self._obs_dim = obs.shape[-1]

        # Action masks
        masks: list[Tensor] = []
        for tl_id in self.tl_ids:
            mask = self.constraints.get_action_mask(
                tl_id,
                self._current_green[tl_id],
                self._elapsed_green[tl_id],
            )
            # Pad to uniform num_actions
            padded = torch.zeros(self.num_actions, dtype=torch.bool)
            padded[: mask.shape[0]] = mask
            masks.append(padded)

        action_mask = torch.stack(masks, dim=0)  # [n_agents, num_actions]

        agents_td = TensorDict(
            {
                "observation": obs,
                "action_mask": action_mask,
            },
            batch_size=[self.n_agents],
        )

        td = TensorDict(
            {
                "agents": agents_td,
                "edge_index": self.edge_index,
            },
            batch_size=[],
        )
        if self.edge_attr is not None:
            td["edge_attr"] = self.edge_attr

        return td

    def _get_observation(self, tl_id: str) -> Tensor:
        """Build observation vector for one intersection.

        Features per controlled lane:
            queue_length, waiting_time, occupancy, mean_speed

        Plus global agent features:
            current_phase (one-hot), elapsed_green (scalar)
        """
        lanes = self._controlled_lanes[tl_id]
        n_lanes = len(lanes)

        queue = torch.tensor(
            [self.adapter.get_lane_halting_number(l) for l in lanes],
            dtype=torch.float32,
        )
        wait = torch.tensor(
            [self.adapter.get_lane_waiting_time(l) for l in lanes],
            dtype=torch.float32,
        )
        occ = torch.tensor(
            [self.adapter.get_lane_occupancy(l) for l in lanes],
            dtype=torch.float32,
        )
        speed = torch.tensor(
            [self.adapter.get_lane_mean_speed(l) for l in lanes],
            dtype=torch.float32,
        )

        # Phase one-hot
        n_green = len(self._green_phases[tl_id])
        phase_idx = self._green_phases[tl_id].index(
            self._current_green[tl_id]
        ) if self._current_green[tl_id] in self._green_phases[tl_id] else 0
        phase_onehot = torch.zeros(n_green, dtype=torch.float32)
        phase_onehot[phase_idx] = 1.0

        elapsed = torch.tensor(
            [self._elapsed_green[tl_id]], dtype=torch.float32
        )

        return torch.cat([queue, wait, occ, speed, phase_onehot, elapsed])

    # ==================================================================
    # Reward
    # ==================================================================
    def _compute_rewards(self) -> Tensor:
        metrics_list: list[IntersectionMetrics] = []
        for tl_id in self.tl_ids:
            lanes = self._controlled_lanes[tl_id]
            m = IntersectionMetrics(
                queue_lengths=[
                    self.adapter.get_lane_halting_number(l) for l in lanes
                ],
                waiting_times=[
                    self.adapter.get_lane_waiting_time(l) for l in lanes
                ],
                mean_speeds=[
                    self.adapter.get_lane_mean_speed(l) for l in lanes
                ],
                occupancies=[
                    self.adapter.get_lane_occupancy(l) for l in lanes
                ],
            )
            metrics_list.append(m)
        return self.reward_calc.compute_batch(metrics_list)

    # ==================================================================
    # Phase helpers
    # ==================================================================
    def _extract_green_phases(self, tl_id: str) -> list[int]:
        """Return indices of phases that are 'green' (contain 'G' or 'g')."""
        logics = self.adapter.get_program_logic(tl_id)
        if not logics:
            return [0]
        phases = logics[0].phases
        green_indices: list[int] = []
        for i, phase in enumerate(phases):
            state = phase.state
            if "G" in state or "g" in state:
                green_indices.append(i)
        return green_indices if green_indices else [0]

    def _get_num_phases(self, tl_id: str) -> int:
        logics = self.adapter.get_program_logic(tl_id)
        if not logics:
            return 1
        return len(logics[0].phases)

    def _apply_transitions(self) -> None:
        """Apply yellow/all-red transitions for agents mid-switch."""
        for tl_id in self.tl_ids:
            done = self.constraints.tick_transition(tl_id, seconds=1)
            dest = self.constraints.get_transition_phase_idx(tl_id)
            if dest is not None and done:
                self.adapter.set_phase(tl_id, dest)
            elif dest is not None:
                # Still in yellow/all-red — SUMO handles the actual phase
                pass
