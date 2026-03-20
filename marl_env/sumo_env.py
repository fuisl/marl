"""Unified RESCO-native SUMO environment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import torch
from tensordict import TensorDict
from torch import Tensor

from marl_env.action_constraints import ActionConstraints
from marl_env.graph_builder import GraphBuilder
from marl_env.observation_adapter import (
    CanonicalObservationLayout,
    GraphMetadata,
    build_canonical_snapshot,
    build_graph_metadata,
)
from marl_env.reward import compute_rewards
from marl_env.resco_metadata import get_resco_map_metadata
from marl_env.resco_observation import RescoSignalState, make_resco_vehicle
from marl_env.resco_reporting import RESCO_RAW_CSV_FIELDS, load_episode_raw_metrics
from marl_env.traci_adapter import TraCIAdapter, tc


class TrafficSignalEnv:
    """Single RESCO-native traffic-signal environment over SUMO."""

    def __init__(
        self,
        net_file: str,
        route_file: str,
        *,
        delta_t: int = 10,
        step_length: int | None = None,
        reward_name: str = "wait",
        yellow_duration: int = 3,
        all_red_duration: int = 2,
        min_green_duration: int = 10,
        sumo_binary: str = "sumo",
        gui: bool = False,
        begin_time: int = 0,
        end_time: int = 3600,
        additional_files: list[str] | None = None,
        timeloss_subscription_policy: str = "strict",
        max_distance: int = 200,
        output_dir: str | None = None,
    ) -> None:
        if step_length is not None:
            delta_t = int(step_length)

        self.net_file = net_file
        self.route_file = route_file
        self.delta_t = int(delta_t)
        self.reward_name = str(reward_name)
        self.max_distance = int(max_distance)
        self.output_dir = None if output_dir in (None, "") else Path(str(output_dir))

        self.adapter = TraCIAdapter(
            net_file=net_file,
            route_file=route_file,
            sumo_binary=sumo_binary,
            gui=gui,
            delta_t=self.delta_t,
            begin_time=begin_time,
            end_time=end_time,
            additional_files=additional_files,
            timeloss_subscription_policy=timeloss_subscription_policy,
        )
        self.constraints = ActionConstraints(
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
            min_green_duration=min_green_duration,
        )

        self.tl_ids: list[str] = []
        self.n_agents: int = 0
        self.graph_builder: GraphBuilder | None = None
        self._graph_metadata: GraphMetadata | None = None

        self._green_phases: dict[str, list[int]] = {}
        self._controlled_lanes: dict[str, list[str]] = {}
        self._elapsed_green: dict[str, float] = {}
        self._current_green: dict[str, int] = {}
        self._pending_target_green: dict[str, int | None] = {}
        self._yellow_phase_map: dict[str, dict[tuple[int, int], int]] = {}
        self._all_red_phase_map: dict[str, dict[tuple[int, int], int]] = {}

        self._episode_index: int = 0
        self._tripinfo_path: Path | None = None
        self._metrics_path: Path | None = None
        self._metrics_rows: list[dict[str, Any]] = []
        self._episode_global_reward: float = 0.0
        self._last_episode_metrics: dict[str, float] = {}

        self._map_metadata: dict[str, Any] = {}
        self._phase_pairs: list[list[str]] = []
        self._signals: dict[str, RescoSignalState] = {}
        self._route_vehicle_count_cache: int | None = None
        self._observation_layout = CanonicalObservationLayout(max_lanes=0)

    # ==================================================================
    # Core interface
    # ==================================================================
    def reset(self) -> TensorDict:
        self.adapter.close()
        self._episode_index += 1
        self._prepare_artifacts()
        self.adapter.start()

        self._metrics_rows = []
        self._episode_global_reward = 0.0
        self._last_episode_metrics = {}

        self.tl_ids = self.adapter.get_traffic_light_ids()
        self.n_agents = len(self.tl_ids)

        self._initialize_graph_metadata()
        self._initialize_phase_state()
        self._initialize_signal_state()
        self._finalize_signal_observations()

        return self._build_tensordict()

    def step(self, actions: Tensor) -> TensorDict:
        for index, tl_id in enumerate(self.tl_ids):
            action_idx = int(actions[index].item())
            target_green = self.constraints.action_to_green_phase(tl_id, action_idx)
            self._maybe_apply_action(tl_id, target_green)

        for _ in range(self.delta_t):
            self.adapter.simulation_step()
            self._sync_phase_state_from_sumo()
            self._collect_context_snapshots()

        self._finalize_signal_observations()
        td = self._build_tensordict()

        reward_map = compute_rewards(
            reward_name=self.reward_name,
            signals=self._signals,
        )
        rewards = torch.tensor(
            [float(reward_map[tl_id]) for tl_id in self.tl_ids],
            dtype=torch.float32,
        )
        self._episode_global_reward += float(rewards.sum().item())
        self._record_metrics_step(rewards)
        td["agents", "reward"] = rewards.unsqueeze(-1)

        done = (
            self.adapter.current_time >= self.adapter.end_time
            or self.adapter.min_expected_vehicles == 0
        )
        td["done"] = torch.tensor([done], dtype=torch.bool)
        td["agents", "done"] = torch.full((self.n_agents, 1), done, dtype=torch.bool)

        if done:
            self._finalize_episode()
        return td

    def close(self) -> None:
        self.adapter.close()

    # ==================================================================
    # Public metadata
    # ==================================================================
    @property
    def num_actions(self) -> int:
        if not self._green_phases:
            return 0
        return max(len(phases) for phases in self._green_phases.values())

    @property
    def observation_dim(self) -> int:
        return int(self._observation_layout.feature_dim)

    @property
    def observation_layout(self) -> CanonicalObservationLayout:
        return self._observation_layout

    def get_observation_layout(self) -> dict[str, int]:
        return self._observation_layout.as_dict()

    def get_signal_specs(self) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        for tl_id in self.tl_ids:
            signal = self._signals[tl_id]
            specs[tl_id] = {
                "signal_id": tl_id,
                "directions": list(signal.directions),
                "phase_pairs": [list(pair) for pair in self._phase_pairs],
                "pair_to_act_map": dict(sorted(signal.pair_to_act_map.items())),
                "local_num_actions": int(signal.local_num_actions),
                "fixed_timings": list(signal.fixed_timings),
                "fixed_phase_order_idx": int(signal.fixed_phase_order_idx),
                "fixed_offset": int(signal.fixed_offset),
                "lane_order": list(signal.lanes),
                "lane_sets": {
                    str(direction): list(lanes)
                    for direction, lanes in signal.lane_sets.items()
                },
                "lane_sets_outbound": {
                    str(direction): list(lanes)
                    for direction, lanes in signal.lane_sets_outbound.items()
                },
                "out_lane_to_signal_id": dict(signal.out_lane_to_signal_id),
                "downstream": dict(signal.downstream),
            }
        return specs

    def get_graph_metadata(self) -> GraphMetadata:
        if self._graph_metadata is None:
            raise RuntimeError("Graph metadata is unavailable before reset().")
        return self._graph_metadata

    def get_artifact_paths(self) -> dict[str, str]:
        paths: dict[str, str] = {}
        if self._tripinfo_path is not None:
            paths["tripinfo"] = str(self._tripinfo_path)
        if self._metrics_path is not None:
            paths["metrics"] = str(self._metrics_path)
        return paths

    def get_episode_metrics(self) -> dict[str, float]:
        return dict(self._last_episode_metrics)

    # Compatibility aliases during the env unification.
    def get_episode_kpis(self) -> dict[str, float]:
        return self.get_episode_metrics()

    def get_benchmark_artifact_paths(self) -> dict[str, str]:
        return self.get_artifact_paths()

    # ==================================================================
    # Tensor helpers
    # ==================================================================
    def _build_tensordict(self) -> TensorDict:
        observations = torch.stack(
            [
                build_canonical_snapshot(
                    signal=self._signals[tl_id],
                    layout=self._observation_layout,
                )
                for tl_id in self.tl_ids
            ],
            dim=0,
        )

        action_masks = torch.stack(
            [self._get_action_mask(tl_id) for tl_id in self.tl_ids],
            dim=0,
        )

        agents_td = TensorDict(
            {
                "observation": observations,
                "action_mask": action_masks,
            },
            batch_size=[self.n_agents],
        )
        return TensorDict({"agents": agents_td}, batch_size=[])

    def _get_action_mask(self, tl_id: str) -> Tensor:
        padded = torch.zeros(self.num_actions, dtype=torch.bool)
        padded[: len(self._green_phases[tl_id])] = True
        return padded

    # ==================================================================
    # Reset helpers
    # ==================================================================
    def _initialize_graph_metadata(self) -> None:
        self.graph_builder = GraphBuilder(
            self.net_file,
            self.tl_ids,
            mode="all_intersections",
        )
        edge_index, edge_attr = self.graph_builder.build()
        self._graph_metadata = build_graph_metadata(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_ids=self.graph_builder.node_ids,
            attached_rl_ids_by_node=self.graph_builder.attached_rl_ids_by_node,
            agent_node_indices=self.graph_builder.agent_node_indices,
            agent_node_mask=self.graph_builder.agent_node_mask,
        )

    def _initialize_phase_state(self) -> None:
        self._green_phases = {}
        self._controlled_lanes = {}
        self._elapsed_green = {}
        self._current_green = {}
        self._pending_target_green = {}
        self._yellow_phase_map = {}
        self._all_red_phase_map = {}

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

            raw_phase = self.adapter.get_phase(tl_id)
            self._current_green[tl_id] = raw_phase if raw_phase in green_phases else green_phases[0]

            self.constraints.register_agent(
                tl_id,
                num_phases=num_phases,
                green_phase_indices=green_phases,
                yellow_phase_map=yellow_map,
                all_red_phase_map=all_red_map,
            )

    def _initialize_signal_state(self) -> None:
        if tc is None:
            raise RuntimeError("traci.constants is required for RESCO-compatible context subscriptions.")

        self._map_metadata = get_resco_map_metadata(net_file=self.net_file)
        self._phase_pairs = [
            [str(pair[0]), str(pair[1])]
            for pair in self._map_metadata["phase_pairs"]
        ]
        self._route_vehicle_count_cache = self._count_route_vehicles(self.route_file)
        self._signals = {}

        vehicle_vars = [
            tc.VAR_LANE_ID,
            tc.VAR_LANEPOSITION,
            tc.VAR_ACCELERATION,
            tc.VAR_SPEED,
            tc.VAR_FUELCONSUMPTION,
            tc.VAR_WAITING_TIME,
            tc.VAR_ALLOWED_SPEED,
            tc.VAR_TYPE,
            tc.VAR_TIMELOSS,
        ]

        for tl_id in self.tl_ids:
            if tl_id not in self._map_metadata:
                raise KeyError(
                    f"Traffic light {tl_id!r} is missing from vendored metadata for {self.net_file!r}."
                )
            signal_meta = dict(self._map_metadata[tl_id])
            pair_to_act_map = {
                int(global_idx): int(local_idx)
                for global_idx, local_idx in signal_meta.get("pair_to_act_map", {}).items()
            }
            if not pair_to_act_map:
                raise ValueError(
                    f"Traffic light {tl_id!r} is missing pair_to_act_map metadata."
                )

            local_actions = sorted(set(pair_to_act_map.values()))
            if local_actions != list(range(len(local_actions))):
                raise ValueError(
                    f"Traffic light {tl_id!r} uses non-contiguous local actions {local_actions}."
                )

            green_phase_count = len(self._green_phases.get(tl_id, []))
            if green_phase_count < len(local_actions):
                raise ValueError(
                    f"Traffic light {tl_id!r} exposes {green_phase_count} green phases, "
                    f"but metadata resolves to {len(local_actions)} local actions."
                )
            if green_phase_count > len(local_actions):
                self._trim_green_phases_to_metadata(
                    tl_id=tl_id,
                    local_action_count=len(local_actions),
                )

            signal_meta["pair_to_act_map"] = pair_to_act_map
            lane_ids: list[str] = []
            for lane_group in signal_meta["lane_sets"].values():
                for lane_id in lane_group:
                    if lane_id not in lane_ids:
                        lane_ids.append(lane_id)

            lane_lengths = {
                lane_id: float(self.adapter.get_lane_length(lane_id))
                for lane_id in lane_ids
            }
            lane_speed_limits = {
                lane_id: float(self.adapter.get_lane_max_speed(lane_id))
                for lane_id in lane_ids
            }
            self._signals[tl_id] = RescoSignalState(
                signal_id=tl_id,
                signal_meta=signal_meta,
                all_signal_meta=self._map_metadata,
                lane_lengths=lane_lengths,
                lane_speed_limits=lane_speed_limits,
            )
            self.adapter.subscribe_junction_context(
                tl_id,
                float(self.max_distance + 25),
                vehicle_vars,
            )

        max_lanes = max((len(signal.lanes) for signal in self._signals.values()), default=0)
        self._observation_layout = CanonicalObservationLayout(max_lanes=max_lanes)

    def _trim_green_phases_to_metadata(self, *, tl_id: str, local_action_count: int) -> None:
        benchmark_green_phases = list(self._green_phases[tl_id][:local_action_count])
        filtered_yellow_map = {
            pair: phase
            for pair, phase in self._yellow_phase_map[tl_id].items()
            if pair[0] in benchmark_green_phases and pair[1] in benchmark_green_phases
        }
        filtered_all_red_map = {
            pair: phase
            for pair, phase in self._all_red_phase_map[tl_id].items()
            if pair[0] in benchmark_green_phases and pair[1] in benchmark_green_phases
        }

        self._green_phases[tl_id] = benchmark_green_phases
        self._yellow_phase_map[tl_id] = filtered_yellow_map
        self._all_red_phase_map[tl_id] = filtered_all_red_map
        self.constraints.register_agent(
            tl_id,
            num_phases=self._get_num_phases(tl_id),
            green_phase_indices=benchmark_green_phases,
            yellow_phase_map=filtered_yellow_map,
            all_red_phase_map=filtered_all_red_map,
        )
        if self._current_green[tl_id] not in benchmark_green_phases:
            self._current_green[tl_id] = benchmark_green_phases[0]
            self._elapsed_green[tl_id] = 0.0
            self._pending_target_green[tl_id] = None
            self.adapter.set_phase(tl_id, benchmark_green_phases[0])

    # ==================================================================
    # Signal observation helpers
    # ==================================================================
    def _collect_context_snapshots(self) -> None:
        if tc is None:
            return
        for tl_id, signal in self._signals.items():
            subscription = self.adapter.get_junction_context_subscription_results(tl_id)
            for veh_id, vehicle in subscription.items():
                veh_lane = vehicle.get(tc.VAR_LANE_ID)
                if veh_lane is None or veh_lane not in signal.observation.lanes:
                    continue
                if str(veh_id).startswith("ghost"):
                    continue

                lane_length = signal.observation.lanes[veh_lane].length
                distance_from_light = lane_length - float(vehicle.get(tc.VAR_LANEPOSITION, 0.0))
                if distance_from_light > self.max_distance:
                    continue

                signal.observation.add_vehicle(
                    make_resco_vehicle(
                        veh_id=str(veh_id),
                        lane_id=str(veh_lane),
                        speed=float(vehicle.get(tc.VAR_SPEED, 0.0)),
                        acceleration=float(vehicle.get(tc.VAR_ACCELERATION, 0.0)),
                        position=float(distance_from_light),
                        allowed_speed=float(vehicle.get(tc.VAR_ALLOWED_SPEED, 0.0)),
                        fuel_consumption=float(vehicle.get(tc.VAR_FUELCONSUMPTION, 0.0)),
                        vehicle_type=str(vehicle.get(tc.VAR_TYPE, "car")),
                    ),
                    step_ratio=1.0,
                )

    def _finalize_signal_observations(self) -> None:
        for tl_id, signal in self._signals.items():
            signal.observation.finalize_step(
                current_phase=self._current_action_index(tl_id),
                phase_length=int(self._elapsed_green.get(tl_id, 0.0)),
            )

    def _current_action_index(self, tl_id: str) -> int:
        current_green = self._current_green.get(tl_id)
        if current_green is None:
            return 0
        if current_green in self._green_phases.get(tl_id, []):
            return self.constraints.green_phase_to_action(tl_id, current_green)
        return 0

    # ==================================================================
    # Metric helpers
    # ==================================================================
    def _prepare_artifacts(self) -> None:
        output_dir = self.output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._tripinfo_path = output_dir / f"tripinfo_{self._episode_index}.xml"
        self._metrics_path = output_dir / f"metrics_{self._episode_index}.csv"
        self.adapter.set_tripinfo_output(str(self._tripinfo_path))

    def _record_metrics_step(self, rewards: Tensor) -> None:
        reward_map = {
            tl_id: float(rewards[index].item())
            for index, tl_id in enumerate(self.tl_ids)
        }
        queue_lengths = {
            tl_id: float(signal.observation.total_queued)
            for tl_id, signal in self._signals.items()
        }
        max_queues = {
            tl_id: float(signal.observation.max_queue)
            for tl_id, signal in self._signals.items()
        }
        vehicles = {
            tl_id: float(self._route_vehicle_count_cache or 0)
            for tl_id in self.tl_ids
        }
        phase_length = {
            tl_id: float(signal.observation.phase_length)
            for tl_id, signal in self._signals.items()
        }
        self._metrics_rows.append(
            {
                "step": max(float(self.adapter.current_time) - 1.0, 0.0),
                "rewards": reward_map,
                "max_queues": max_queues,
                "queue_lengths": queue_lengths,
                "vehicles": vehicles,
                "phase_length": phase_length,
            }
        )

    def _finalize_episode(self) -> None:
        if self._metrics_path is None or self._tripinfo_path is None:
            self.adapter.close()
            self._last_episode_metrics = {
                "global_reward": float(self._episode_global_reward),
            }
            return

        self._write_metrics_csv(self._metrics_path)
        self.adapter.close()
        if (not self._tripinfo_path.exists()) or (not self._metrics_path.exists()):
            self._last_episode_metrics = {
                "duration": 0.0,
                "waitingTime": 0.0,
                "timeLoss": 0.0,
                "rewards": 0.0,
                "max_queues": 0.0,
                "queue_lengths": 0.0,
                "vehicles": 0.0,
                "phase_length": 0.0,
                "global_reward": float(self._episode_global_reward),
            }
            return

        self._last_episode_metrics = load_episode_raw_metrics(
            tripinfo_path=self._tripinfo_path,
            metrics_path=self._metrics_path,
            global_reward=self._episode_global_reward,
        )

    def _write_metrics_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RESCO_RAW_CSV_FIELDS))
            writer.writeheader()
            for row in self._metrics_rows:
                writer.writerow(
                    {
                        metric: str(row[metric])
                        for metric in RESCO_RAW_CSV_FIELDS
                    }
                )

    @staticmethod
    def _count_route_vehicles(route_file: str) -> int:
        total = 0
        for route_path in str(route_file).split(","):
            candidate = Path(route_path.strip())
            if not route_path.strip():
                continue
            try:
                xml_root = ET.parse(candidate).getroot()
            except Exception:
                continue

            for child in xml_root:
                tag = child.tag.split("}")[-1]
                if tag in {"vehicle", "trip"}:
                    total += 1
                    continue
                if tag != "flow":
                    continue
                if "number" in child.attrib:
                    total += int(float(child.attrib["number"]))
                    continue
                begin = float(child.attrib.get("begin", 0.0))
                end = float(child.attrib.get("end", begin))
                duration = max(end - begin, 0.0)
                if "vehsPerHour" in child.attrib:
                    total += int(duration * float(child.attrib["vehsPerHour"]) / 3600.0)
                elif "probability" in child.attrib:
                    total += int(duration * float(child.attrib["probability"]))
        return total

    # ==================================================================
    # Phase helpers
    # ==================================================================
    def _extract_green_phases(self, tl_id: str) -> list[int]:
        logics = self.adapter.get_program_logic(tl_id)
        if not logics:
            return [0]
        phases = logics[0].phases
        green_indices: list[int] = []
        for index, phase in enumerate(phases):
            state = phase.state
            if "y" in state or "Y" in state:
                continue
            if state.count("r") + state.count("R") + state.count("s") + state.count("S") == len(state):
                continue
            if "G" in state or "g" in state:
                green_indices.append(index)
        return green_indices if green_indices else [0]

    def _get_num_phases(self, tl_id: str) -> int:
        logics = self.adapter.get_program_logic(tl_id)
        if not logics:
            return 1
        return len(logics[0].phases)

    def _sync_phase_state_from_sumo(self) -> None:
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

    def _maybe_apply_action(self, tl_id: str, target_green: int) -> None:
        raw_phase = self.adapter.get_phase(tl_id)
        if raw_phase not in self._green_phases[tl_id]:
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

    def _build_transition_maps(
        self,
        tl_id: str,
        num_phases: int,
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        logics = self.adapter.get_program_logic(tl_id)
        if not logics:
            return {}, {}

        phases = logics[0].phases
        green_phases = self._green_phases.get(tl_id) or self._extract_green_phases(tl_id)

        yellow_map: dict[tuple[int, int], int] = {}
        all_red_map: dict[tuple[int, int], int] = {}

        for index in range(len(green_phases)):
            from_green = green_phases[index]
            to_green = green_phases[(index + 1) % len(green_phases)]
            phase_index = (from_green + 1) % num_phases
            while phase_index != to_green:
                state = phases[phase_index].state
                if "y" in state or "Y" in state:
                    yellow_map.setdefault((from_green, to_green), phase_index)
                elif all(char in ("r", "R") for char in state):
                    all_red_map.setdefault((from_green, to_green), phase_index)
                phase_index = (phase_index + 1) % num_phases

        return yellow_map, all_red_map
