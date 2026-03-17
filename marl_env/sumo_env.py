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
from marl_env.traci_adapter import TraCIAdapter, _VAR_ACCUM_WAITING, _VAR_TIME_LOSS


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
        graph_builder_mode: str = "original",
    ) -> None:
        self.delta_t = delta_t
        self.net_file = net_file
        self.graph_builder_mode = graph_builder_mode

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
        self.graph_node_ids: list[str] = []
        self.agent_node_indices: Tensor | None = None
        self.agent_node_mask: Tensor | None = None

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
        self._last_interval_flow: dict[str, float] = {
            "arrived_vehicles": 0.0,
            "departed_vehicles": 0.0,
            "teleported_vehicles": 0.0,
        }
        self._depart_time_by_vehicle: dict[str, float] = {}
        self._episode_travel_time_sum_s: float = 0.0
        self._episode_arrived_vehicles: int = 0
        # Per-vehicle benchmark accumulators (NeurIPS RESCO definitions)
        self._episode_wait_time_sum_s: float = 0.0   # accumulated waiting per completed veh
        self._episode_time_loss_sum_s: float = 0.0   # timeLoss per completed veh (= avg delay)
        # Time-averaged queue: sum of halting vehicles across all sim steps
        self._sim_step_queue_sum: float = 0.0
        self._sim_step_count: int = 0
        self._all_controlled_lanes: set[str] = set()
        self._node_incoming_lanes: dict[str, list[str]] = {}
        self._node_attached_rl_ids: dict[str, tuple[str, ...]] = {}

    # ==================================================================
    # Core gym-like interface
    # ==================================================================
    def reset(self) -> TensorDict:
        """Start a new episode. Returns initial observation ``TensorDict``."""
        self.adapter.close()  # no-op on first call
        self.adapter.start()
        self._depart_time_by_vehicle = {}
        self._episode_travel_time_sum_s = 0.0
        self._episode_arrived_vehicles = 0
        self._episode_wait_time_sum_s = 0.0
        self._episode_time_loss_sum_s = 0.0
        self._sim_step_queue_sum = 0.0
        self._sim_step_count = 0

        # Discover agents
        self.tl_ids = self.adapter.get_traffic_light_ids()
        self.n_agents = len(self.tl_ids)

        # Build static graph once
        self.graph_builder = GraphBuilder(
            self.net_file,
            self.tl_ids,
            mode=self.graph_builder_mode,
        )
        self.edge_index, self.edge_attr = self.graph_builder.build()
        self.graph_node_ids = self.graph_builder.node_ids
        self.agent_node_indices = self.graph_builder.agent_node_indices
        self.agent_node_mask = self.graph_builder.agent_node_mask
        self._node_incoming_lanes = self._build_graph_node_incoming_lane_map()
        self._node_attached_rl_ids = {
            node_id: attached
            for node_id, attached in zip(
                self.graph_node_ids,
                self.graph_builder.attached_rl_ids_by_node,
                strict=True,
            )
        }

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
        lane_counts = [len(self._controlled_lanes[tl]) for tl in self.tl_ids]
        if self.graph_builder_mode == "all_intersections":
            lane_counts.extend(len(lanes) for lanes in self._node_incoming_lanes.values())
        self._max_lanes = max(lane_counts) if lane_counts else 0
        self._max_green = max(
            len(self._green_phases[tl]) for tl in self.tl_ids
        )

        # Cache the full set of controlled lanes once per episode for O(1) queue tracking.
        self._all_controlled_lanes = {
            lane for lanes in self._controlled_lanes.values() for lane in lanes
        }

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
        arrived_total = 0.0
        departed_total = 0.0
        teleported_total = 0.0
        for _ in range(self.delta_t):
            self.adapter.simulation_step()
            self._sync_phase_state_from_sumo()
            arrived_total += float(self.adapter.get_arrived_number())
            departed_total += float(self.adapter.get_departed_number())
            teleported_total += float(self.adapter.get_teleported_number())

            now = float(self.adapter.current_time)
            for vid in self.adapter.get_departed_ids():
                # Record departure time and subscribe to per-vehicle stats.
                self._depart_time_by_vehicle.setdefault(vid, now)
                self.adapter.subscribe_vehicle(vid)

            # Accumulate time-averaged queue: total halting vehicles this sim step.
            self._sim_step_queue_sum += float(
                sum(self.adapter.get_lane_halting_number(l) for l in self._all_controlled_lanes)
            )
            self._sim_step_count += 1

            for vid in self.adapter.get_arrived_ids():
                t_depart = self._depart_time_by_vehicle.pop(vid, None)
                if t_depart is None:
                    continue
                self._episode_travel_time_sum_s += max(0.0, now - t_depart)
                self._episode_arrived_vehicles += 1
                # Subscription results stay readable for one step after arrival.
                sub = self.adapter.get_vehicle_subscription_results(vid)
                self._episode_wait_time_sum_s += float(sub.get(_VAR_ACCUM_WAITING, 0.0))
                self._episode_time_loss_sum_s += float(sub.get(_VAR_TIME_LOSS, 0.0))

        self._last_interval_flow = {
            "arrived_vehicles": arrived_total,
            "departed_vehicles": departed_total,
            "teleported_vehicles": teleported_total,
        }

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

    def get_interval_kpis(self) -> dict[str, float]:
        """Return network-level KPIs for the latest decision interval."""
        lane_ids: set[str] = set()
        for lanes in self._controlled_lanes.values():
            lane_ids.update(lanes)

        if not lane_ids:
            return {
                "avg_delay_s": 0.0,
                "avg_queue_length": 0.0,
                "avg_speed_mps": 0.0,
                "avg_occupancy_pct": 0.0,
                "min_expected_vehicles": float(self.adapter.min_expected_vehicles),
                **self._last_interval_flow,
            }

        lane_count = float(len(lane_ids))
        total_waiting = sum(self.adapter.get_lane_waiting_time(l) for l in lane_ids)
        total_vehicles = sum(self.adapter.get_lane_vehicle_count(l) for l in lane_ids)

        # Per-vehicle delay is more stable across different network sizes.
        avg_waiting = total_waiting / max(float(total_vehicles), 1.0)
        avg_queue = sum(self.adapter.get_lane_halting_number(l) for l in lane_ids) / lane_count
        avg_speed = sum(self.adapter.get_lane_mean_speed(l) for l in lane_ids) / lane_count
        avg_occupancy = sum(self.adapter.get_lane_occupancy(l) for l in lane_ids) / lane_count

        return {
            "avg_delay_s": float(avg_waiting),
            "avg_queue_length": float(avg_queue),
            "avg_speed_mps": float(avg_speed),
            "avg_occupancy_pct": float(avg_occupancy),
            "min_expected_vehicles": float(self.adapter.min_expected_vehicles),
            "network_total_waiting_s": float(total_waiting),
            "network_total_vehicles": float(total_vehicles),
            **self._last_interval_flow,
        }

    def get_episode_kpis(self) -> dict[str, float]:
        """Return episode-level KPIs matching the NeurIPS RESCO benchmark definitions.

        All per-vehicle metrics are averaged over completed vehicles only
        (vehicles still in the network at episode end are excluded).
        Queue is time-averaged over simulation steps.

        Keys
        ----
        avg_travel_time_s : per-completed-vehicle (exit - enter)
        avg_wait_s        : per-completed-vehicle accumulated waiting time
                            (time with speed < 0.1 m/s)
        avg_delay_s       : per-completed-vehicle timeLoss
                            (= trip_time - free-flow_time)
        avg_queue_length  : mean over sim steps of total halting vehicles
        arrived_vehicles  : number of vehicles counted
        """
        n = self._episode_arrived_vehicles
        avg_travel_time = self._episode_travel_time_sum_s / n if n > 0 else 0.0
        avg_wait = self._episode_wait_time_sum_s / n if n > 0 else 0.0
        avg_delay = self._episode_time_loss_sum_s / n if n > 0 else 0.0
        avg_queue = self._sim_step_queue_sum / max(self._sim_step_count, 1)
        return {
            "avg_travel_time_s": float(avg_travel_time),
            "avg_wait_s": float(avg_wait),
            "avg_delay_s": float(avg_delay),
            "avg_queue_length": float(avg_queue),
            "arrived_vehicles": float(n),
        }

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
        graph_obs = self._build_graph_observation(obs)

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
                "graph_observation": graph_obs,
                "agent_node_indices": self.agent_node_indices,
                "agent_node_mask": self.agent_node_mask,
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

    def _build_observation_from_lanes_and_phase(
        self,
        lanes: list[str],
        *,
        phase_owner_tl_id: str | None,
    ) -> Tensor:
        """Build one observation vector from lane metrics and optional phase state.

        Layout (padded to uniform size)::

            [queue(max_lanes) | wait(max_lanes) | occ(max_lanes)
             | speed(max_lanes) | phase_onehot(max_green) | elapsed(1)]
        """
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
        phase_onehot = torch.zeros(self._max_green, dtype=torch.float32)
        elapsed_value = 0.0
        if phase_owner_tl_id is not None:
            current_gp = self._current_green[phase_owner_tl_id]
            if current_gp in self._green_phases[phase_owner_tl_id]:
                action_idx = self.constraints.green_phase_to_action(
                    phase_owner_tl_id,
                    current_gp,
                )
            else:
                action_idx = 0
            phase_onehot[action_idx] = 1.0
            elapsed_value = self._elapsed_green[phase_owner_tl_id]

        elapsed = torch.tensor([elapsed_value], dtype=torch.float32)

        return torch.cat([queue, wait, occ, speed, phase_onehot, elapsed])

    def _get_observation(self, tl_id: str) -> Tensor:
        """Build observation vector for one controlled traffic light."""
        return self._build_observation_from_lanes_and_phase(
            self._controlled_lanes[tl_id],
            phase_owner_tl_id=tl_id,
        )

    def _build_graph_observation(self, agent_obs: Tensor) -> Tensor:
        if self.graph_builder_mode != "all_intersections":
            return agent_obs.clone()

        graph_obs_list = [
            self._get_graph_node_observation(node_id)
            for node_id in self.graph_node_ids
        ]
        return torch.stack(graph_obs_list, dim=0)

    def _get_graph_node_observation(self, node_id: str) -> Tensor:
        attached_rl_ids = self._node_attached_rl_ids.get(node_id, ())
        phase_owner = attached_rl_ids[0] if attached_rl_ids else None
        lanes = self._node_incoming_lanes.get(node_id, [])
        return self._build_observation_from_lanes_and_phase(
            lanes,
            phase_owner_tl_id=phase_owner,
        )

    def _build_graph_node_incoming_lane_map(self) -> dict[str, list[str]]:
        if self.graph_builder is None or self.graph_builder_mode != "all_intersections":
            return {}

        node_lanes: dict[str, list[str]] = {}
        for node_id in self.graph_node_ids:
            try:
                node = self.graph_builder.net.getNode(node_id)
            except KeyError:
                node_lanes[node_id] = []
                continue

            lane_ids: list[str] = []
            for edge in node.getIncoming():
                if hasattr(edge, "getLanes"):
                    lane_ids.extend(lane.getID() for lane in edge.getLanes())
            node_lanes[node_id] = lane_ids

        return node_lanes

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
                pressure=self._compute_intersection_pressure(tl_id, lanes),
            )
            metrics_list.append(m)
        return self.reward_calc.compute_batch(metrics_list)

    def _compute_intersection_pressure(self, tl_id: str, incoming_lanes: list[str]) -> float:
        """Compute queue pressure = incoming halting - outgoing halting.

        Outgoing lanes are inferred from controlled links and deduplicated.
        """
        in_queue = sum(self.adapter.get_lane_halting_number(l) for l in incoming_lanes)

        outgoing_lanes: set[str] = set()
        for signal_links in self.adapter.get_controlled_links(tl_id):
            for link in signal_links:
                # Each link tuple is (in_lane, out_lane, via_lane)
                if len(link) >= 2 and link[1]:
                    outgoing_lanes.add(link[1])

        out_queue = sum(self.adapter.get_lane_halting_number(l) for l in outgoing_lanes)
        return float(in_queue - out_queue)

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
