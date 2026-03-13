"""Custom SUMO multi-agent traffic-signal environment.

Outputs TorchRL-style ``TensorDict`` with shared graph tensors at the top
level and per-agent tensors nested under ``"agents"``.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

from marl_env.action_constraints import ActionConstraints
from marl_env.graph_builder import GraphBuilder
from marl_env.reward import IntersectionMetrics, RewardCalculator
from marl_env.traci_adapter import TraCIAdapter


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
        self._pending_target_green: dict[str, int | None] = {}
        self._yellow_phase_map: dict[str, dict[tuple[int, int], int]] = {}
        self._all_red_phase_map: dict[str, dict[tuple[int, int], int]] = {}
        self._max_lanes: int = 0
        self._max_green: int = 0

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
            num_phases = self._get_num_phases(tl_id)
            green_phases = self._extract_green_phases(tl_id)
            yellow_map, all_red_map = self._build_transition_maps(tl_id, num_phases)

            self._green_phases[tl_id] = green_phases
            self._controlled_lanes[tl_id] = self.adapter.get_controlled_lanes(tl_id)
            self._elapsed_green[tl_id] = 0.0
            self._pending_target_green[tl_id] = None
            self._yellow_phase_map[tl_id] = yellow_map
            self._all_red_phase_map[tl_id] = all_red_map

            # Snap initial phase to the nearest controllable green
            raw_phase = self.adapter.get_phase(tl_id)
            if raw_phase in green_phases:
                self._current_green[tl_id] = raw_phase
            else:
                self._current_green[tl_id] = green_phases[0]

            self.constraints.register_agent(
                tl_id,
                num_phases=num_phases,
                green_phase_indices=green_phases,
                yellow_phase_map=yellow_map,
                all_red_phase_map=all_red_map,
            )

        # Fixed observation dimensions (for padding)
        self._max_lanes = max(
            len(self._controlled_lanes[tl]) for tl in self.tl_ids
        )
        self._max_green = max(
            len(self._green_phases[tl]) for tl in self.tl_ids
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
        # --- Apply action intents by triggering transition start only ---
        for i, tl_id in enumerate(self.tl_ids):
            action_idx = int(actions[i].item())
            target_green = self.constraints.action_to_green_phase(tl_id, action_idx)
            self._maybe_apply_action(tl_id, target_green)

        # --- Advance simulation by delta_t seconds ---
        for _ in range(self.delta_t):
            self.adapter.simulation_step()
            self._sync_phase_state_from_sumo()

        # --- Build output ---
        td = self._build_tensordict()

        # Rewards
        rewards = self._compute_rewards()
        td["agents", "reward"] = rewards.unsqueeze(-1)  # [n_agents, 1]

        # Done
        done = (
            self.adapter.current_time >= self.adapter.end_time
            or self.adapter.min_expected_vehicles == 0
        )
        td["done"] = torch.tensor([done], dtype=torch.bool)
        td["agents", "done"] = torch.full((self.n_agents, 1), done, dtype=torch.bool)

        return td

    def close(self) -> None:
        self.adapter.close()

    # ==================================================================
    # Observation helpers
    # ==================================================================
    @property
    def observation_dim(self) -> int:
        """Dimension of per-agent observation vector.

        ``4 * max_lanes + max_green + 1``
        (queue + wait + occupancy + speed, per lane, padded)
        (phase one-hot, padded) + elapsed green.
        """
        return 4 * self._max_lanes + self._max_green + 1

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

        # Action masks
        masks: list[Tensor] = []
        for tl_id in self.tl_ids:
            mask = self._get_action_mask(tl_id)
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

    # ==================================================================
    # Observation helpers
    # ==================================================================
    @staticmethod
    def _pad_1d(values: list[float], target_len: int) -> Tensor:
        """Pad or truncate a list of floats to ``target_len``."""
        t = torch.tensor(values, dtype=torch.float32)
        if t.numel() < target_len:
            t = torch.cat(
                [t, torch.zeros(target_len - t.numel(), dtype=torch.float32)]
            )
        else:
            t = t[:target_len]
        return t

    def _get_observation(self, tl_id: str) -> Tensor:
        """Build observation vector for one intersection.

        Layout (padded to uniform size)::

            [queue(max_lanes) | wait(max_lanes) | occ(max_lanes)
             | speed(max_lanes) | phase_onehot(max_green) | elapsed(1)]
        """
        lanes = self._controlled_lanes[tl_id]
        ml = self._max_lanes

        queue = self._pad_1d(
            [self.adapter.get_lane_halting_number(l) for l in lanes], ml
        )
        wait = self._pad_1d(
            [self.adapter.get_lane_waiting_time(l) for l in lanes], ml
        )
        occ = self._pad_1d(
            [self.adapter.get_lane_occupancy(l) for l in lanes], ml
        )
        speed = self._pad_1d(
            [self.adapter.get_lane_mean_speed(l) for l in lanes], ml
        )

        # Phase one-hot (padded to max_green)
        current_gp = self._current_green[tl_id]
        if current_gp in self._green_phases[tl_id]:
            action_idx = self.constraints.green_phase_to_action(tl_id, current_gp)
        else:
            action_idx = 0
        phase_onehot = torch.zeros(self._max_green, dtype=torch.float32)
        phase_onehot[action_idx] = 1.0

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
                queue_lengths=[self.adapter.get_lane_halting_number(l) for l in lanes],
                waiting_times=[self.adapter.get_lane_waiting_time(l) for l in lanes],
                mean_speeds=[self.adapter.get_lane_mean_speed(l) for l in lanes],
                occupancies=[self.adapter.get_lane_occupancy(l) for l in lanes],
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

    def _sync_phase_state_from_sumo(self) -> None:
        """Synchronize cached phase state from actual SUMO phase each second."""
        for tl_id in self.tl_ids:
            raw_phase = self.adapter.get_phase(tl_id)
            green_phases = self._green_phases[tl_id]

            if raw_phase in green_phases:
                if raw_phase != self._current_green[tl_id]:
                    self._current_green[tl_id] = raw_phase
                    self._elapsed_green[tl_id] = 0.0
                else:
                    self._elapsed_green[tl_id] += 1.0

                if self._pending_target_green[tl_id] == raw_phase:
                    self._pending_target_green[tl_id] = None
            else:
                # Transitional phases (yellow/all-red) are managed by SUMO.
                # Keep last known green and elapsed-green timer unchanged.
                pass

    def _get_action_mask(self, tl_id: str) -> Tensor:
        """Return legal action mask based on observed SUMO state."""
        n_actions = len(self._green_phases[tl_id])
        mask = torch.ones(n_actions, dtype=torch.bool)

        raw_phase = self.adapter.get_phase(tl_id)
        if raw_phase not in self._green_phases[tl_id]:
            mask[:] = False
            target = self._pending_target_green[tl_id]
            if target is None:
                target = self._current_green[tl_id]
            if target in self._green_phases[tl_id]:
                keep_action = self.constraints.green_phase_to_action(tl_id, target)
                mask[keep_action] = True
            return mask

        if self._elapsed_green[tl_id] < self.constraints.min_green_duration:
            mask[:] = False
            keep_action = self.constraints.green_phase_to_action(
                tl_id, self._current_green[tl_id]
            )
            mask[keep_action] = True

        return mask

    def _maybe_apply_action(self, tl_id: str, target_green: int) -> None:
        """Trigger a transition start and let SUMO handle subsequent progression."""
        raw_phase = self.adapter.get_phase(tl_id)

        # Ignore new requests while SUMO is in transitional phases.
        if raw_phase not in self._green_phases[tl_id]:
            return

        if self._elapsed_green[tl_id] < self.constraints.min_green_duration:
            return

        current_green = self._current_green[tl_id]
        if target_green == current_green:
            return

        pair = (current_green, target_green)
        start_phase = self._yellow_phase_map[tl_id].get(pair)
        if start_phase is None:
            start_phase = self._all_red_phase_map[tl_id].get(pair)
        if start_phase is None:
            start_phase = target_green

        self._pending_target_green[tl_id] = target_green
        self.adapter.set_phase(tl_id, start_phase)

    def _build_transition_maps(self, tl_id: str, num_phases: int) -> tuple[
        dict[tuple[int, int], int],
        dict[tuple[int, int], int],
    ]:
        """Infer yellow and all-red phase mappings from the SUMO program.

        Heuristic: for each pair of consecutive green phases in the program,
        phases between them that contain 'y' are yellow, and phases that are
        all 'r' are all-red.

        Returns ``(yellow_phase_map, all_red_phase_map)``.
        """
        logics = self.adapter.get_program_logic(tl_id)
        if not logics:
            return {}, {}

        phases = logics[0].phases
        green_phases = self._green_phases.get(tl_id) or self._extract_green_phases(
            tl_id
        )

        yellow_map: dict[tuple[int, int], int] = {}
        all_red_map: dict[tuple[int, int], int] = {}

        # Walk through the program and associate intermediate phases
        for gi in range(len(green_phases)):
            from_gp = green_phases[gi]
            to_gp = green_phases[(gi + 1) % len(green_phases)]

            # Scan phases between from_gp and to_gp (wrapping around)
            idx = (from_gp + 1) % num_phases
            while idx != to_gp:
                state = phases[idx].state
                if "y" in state or "Y" in state:
                    yellow_map.setdefault((from_gp, to_gp), idx)
                elif all(c in ("r", "R") for c in state):
                    all_red_map.setdefault((from_gp, to_gp), idx)
                idx = (idx + 1) % num_phases

        return yellow_map, all_red_map
