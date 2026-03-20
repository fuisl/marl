"""RESCO-compatible signal observations, states, and rewards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RescoVehicle:
    veh_id: str
    lane_id: str
    speed: float
    acceleration: float
    position: float
    allowed_speed: float
    fuel_consumption: float
    vehicle_type: str
    queued: bool = False
    wait: float = 0.0
    delay: float = 0.0
    total_speed: float = 0.0
    total_acceleration: float = 0.0
    times_observed: int = 1
    times_observed_last_step: int = 0
    length: float = 5.0
    min_gap: float = 2.5

    def __post_init__(self) -> None:
        self.total_speed = float(self.speed)
        self.total_acceleration = float(self.acceleration)
        self.delay = self._compute_delay()

    def _compute_delay(self) -> float:
        if self.allowed_speed <= 0.0:
            return 0.0
        return (self.speed - self.allowed_speed) / self.allowed_speed

    @property
    def average_speed(self) -> float:
        return self.total_speed / max(self.times_observed, 1)

    def observe(self, other: "RescoVehicle", step_ratio: float = 1.0) -> None:
        self.times_observed += 1
        self.speed = other.speed
        self.acceleration = other.acceleration
        self.position = other.position
        self.allowed_speed = other.allowed_speed
        self.vehicle_type = other.vehicle_type
        self.fuel_consumption += other.fuel_consumption

        if other.speed < 0.1:
            self.queued = True
        if self.queued:
            self.wait += step_ratio

        self.delay += other._compute_delay()
        self.total_speed += other.speed
        self.total_acceleration += other.acceleration


@dataclass
class RescoLaneObservation:
    lane_id: str
    max_speed: float
    length: float
    vehicles: dict[str, RescoVehicle] = field(default_factory=dict)
    vehicle_count: int = 0
    queued: int = 0
    arrived: int = 0
    departed: int = 0
    max_wait: float = 0.0

    @property
    def approaching(self) -> int:
        return len(self.vehicles) - self.queued

    def add_vehicle(self, vehicle: RescoVehicle, step_ratio: float = 1.0) -> None:
        existing = self.vehicles.get(vehicle.veh_id)
        if existing is None:
            self.vehicles[vehicle.veh_id] = vehicle
            return
        existing.observe(vehicle, step_ratio=step_ratio)


@dataclass
class RescoSignalObservation:
    lane_lengths: dict[str, float]
    lane_speed_limits: dict[str, float]
    lanes: dict[str, RescoLaneObservation] = field(init=False)
    current_phase: int = 0
    phase_length: int = 0
    vehicle_count: int = 0
    departed: int = 0
    arrived: int = 0
    total_wait: float = 0.0
    total_queued: int = 0
    max_queue: int = 0

    def __post_init__(self) -> None:
        self.lanes = {
            lane_id: RescoLaneObservation(
                lane_id=lane_id,
                max_speed=float(self.lane_speed_limits[lane_id]),
                length=float(self.lane_lengths[lane_id]),
            )
            for lane_id in self.lane_lengths
        }

    def add_vehicle(self, vehicle: RescoVehicle, step_ratio: float = 1.0) -> None:
        lane = self.lanes.get(vehicle.lane_id)
        if lane is None:
            return
        lane.add_vehicle(vehicle, step_ratio=step_ratio)

    def finalize_step(self, *, current_phase: int, phase_length: int) -> None:
        self.current_phase = current_phase
        self.phase_length = phase_length
        self.departed = 0
        self.total_wait = 0.0
        self.total_queued = 0
        self.max_queue = 0

        total_vehicle_count = 0
        for lane in self.lanes.values():
            lane.queued = 0
            lane.departed = 0
            lane.arrived = 0
            lane.max_wait = 0.0
            lane_vehicle_count = 0
            pending_removal: list[str] = []

            for vehicle in lane.vehicles.values():
                if vehicle.times_observed_last_step == vehicle.times_observed:
                    self.departed += 1
                    lane.departed += 1
                    pending_removal.append(vehicle.veh_id)
                else:
                    lane_vehicle_count += 1
                    if vehicle.queued:
                        self.total_queued += 1
                        lane.queued += 1
                    self.total_wait += vehicle.wait
                    lane.max_wait = max(lane.max_wait, vehicle.wait)
                vehicle.times_observed_last_step = vehicle.times_observed

            previous_vehicle_count = lane.vehicle_count
            lane.arrived = lane_vehicle_count - (previous_vehicle_count - lane.departed)
            lane.vehicle_count = lane_vehicle_count
            total_vehicle_count += lane_vehicle_count
            self.max_queue = max(self.max_queue, lane.queued)

            for veh_id in pending_removal:
                lane.vehicles.pop(veh_id, None)

        self.arrived = total_vehicle_count - (self.vehicle_count - self.departed)
        self.vehicle_count = total_vehicle_count

    def get_lane(self, lane_id: str) -> RescoLaneObservation:
        return self.lanes[lane_id]


class RescoSignalState:
    """Local RESCO-style signal metadata and rolling vehicle observation."""

    def __init__(
        self,
        *,
        signal_id: str,
        signal_meta: dict[str, Any],
        all_signal_meta: dict[str, Any],
        lane_lengths: dict[str, float],
        lane_speed_limits: dict[str, float],
    ) -> None:
        self.signal_id = signal_id
        self.signal_meta = signal_meta
        self.all_signal_meta = all_signal_meta
        self.lane_sets: dict[str, list[str]] = {
            str(key): list(value) for key, value in signal_meta["lane_sets"].items()
        }
        self.downstream: dict[str, str | None] = {
            str(key): (None if value is None else str(value))
            for key, value in signal_meta["downstream"].items()
        }
        self.fixed_timings = [int(x) for x in signal_meta.get("fixed_timings", [])]
        self.fixed_phase_order_idx = int(signal_meta.get("fixed_phase_order_idx", 0))
        self.fixed_offset = int(signal_meta.get("fixed_offset", 0))
        self.pair_to_act_map: dict[int, int] = {
            int(global_idx): int(local_idx)
            for global_idx, local_idx in signal_meta.get("pair_to_act_map", {}).items()
        }
        self.local_num_actions = len(set(self.pair_to_act_map.values()))

        self.lanes: list[str] = []
        self.outbound_lanes: list[str] = []
        self.inbounds_fr_direction: dict[str, list[str]] = {}
        self.out_lane_to_signal_id: dict[str, str] = {}
        self.lane_sets_outbound: dict[str, list[str]] = {
            direction: [] for direction in self.lane_sets
        }

        self._find_neighbors()
        self.observation = RescoSignalObservation(
            lane_lengths={lane_id: lane_lengths[lane_id] for lane_id in self.lanes},
            lane_speed_limits={
                lane_id: lane_speed_limits[lane_id] for lane_id in self.lanes
            },
        )

    @property
    def directions(self) -> list[str]:
        return list(self.lane_sets)

    def _find_neighbors(self) -> None:
        reversed_directions = {"N": "S", "E": "W", "S": "N", "W": "E"}

        for direction, lane_group in self.lane_sets.items():
            inbound_to_direction = direction.split("-")[0]
            inbound_fr_direction = reversed_directions[inbound_to_direction]
            for lane_id in lane_group:
                if lane_id not in self.lanes:
                    self.lanes.append(lane_id)
                self.inbounds_fr_direction.setdefault(inbound_fr_direction, [])
                if lane_id not in self.inbounds_fr_direction[inbound_fr_direction]:
                    self.inbounds_fr_direction[inbound_fr_direction].append(lane_id)

        for direction, dwn_signal in self.downstream.items():
            if dwn_signal is None:
                continue
            dwn_lane_sets = self.all_signal_meta[dwn_signal]["lane_sets"]
            for key, dwn_lane_group in dwn_lane_sets.items():
                if key.split("-")[0] != direction:
                    continue
                for lane_id in dwn_lane_group:
                    if lane_id not in self.outbound_lanes:
                        self.outbound_lanes.append(lane_id)
                    self.out_lane_to_signal_id[lane_id] = dwn_signal
                for self_key in self.lane_sets:
                    if self_key.split("-")[1] == key.split("-")[0]:
                        self.lane_sets_outbound[self_key].extend(dwn_lane_group)

        for key in self.lane_sets_outbound:
            self.lane_sets_outbound[key] = list(dict.fromkeys(self.lane_sets_outbound[key]))


def make_resco_vehicle(
    *,
    veh_id: str,
    lane_id: str,
    speed: float,
    acceleration: float,
    position: float,
    allowed_speed: float,
    fuel_consumption: float,
    vehicle_type: str,
) -> RescoVehicle:
    return RescoVehicle(
        veh_id=veh_id,
        lane_id=lane_id,
        speed=float(speed),
        acceleration=float(acceleration),
        position=float(position),
        allowed_speed=float(allowed_speed),
        fuel_consumption=float(fuel_consumption),
        vehicle_type=str(vehicle_type),
    )


def build_wave_states(signals: dict[str, RescoSignalState]) -> dict[str, list[float]]:
    states: dict[str, list[float]] = {}
    for signal_id, signal in signals.items():
        state: list[float] = []
        for direction in signal.directions:
            total = 0.0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                total += float(lane.queued + lane.approaching)
            state.append(total)
        states[signal_id] = state
    return states


def build_mplight_states(
    signals: dict[str, RescoSignalState],
    *,
    current_action_by_signal: dict[str, int],
) -> dict[str, list[float]]:
    observations: dict[str, list[float]] = {}
    for signal_id, signal in signals.items():
        obs: list[float] = [float(current_action_by_signal.get(signal_id, 0))]
        for direction in signal.directions:
            pressure = 0.0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                pressure += float(lane.queued)
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id.get(lane_id)
                if dwn_signal in signals:
                    lane = signals[dwn_signal].observation.get_lane(lane_id)
                    pressure -= float(lane.queued)
            obs.append(pressure)
        observations[signal_id] = obs
    return observations


def build_drq_states(
    signals: dict[str, RescoSignalState],
    *,
    current_action_by_signal: dict[str, int],
) -> dict[str, list[float]]:
    observations: dict[str, list[float]] = {}
    for signal_id, signal in signals.items():
        current_phase = int(current_action_by_signal.get(signal_id, 0))
        obs: list[float] = []
        for lane_idx, lane_id in enumerate(signal.lanes):
            lane_obs = signal.observation.get_lane(lane_id)
            total_wait = 0.0
            total_speed = 0.0
            for vehicle in lane_obs.vehicles.values():
                total_wait += float(vehicle.wait)
                total_speed += float(vehicle.average_speed)

            obs.extend(
                [
                    1.0 if lane_idx == current_phase else 0.0,
                    float(lane_obs.approaching),
                    total_wait,
                    float(lane_obs.queued),
                    total_speed,
                ]
            )
        observations[signal_id] = obs
    return observations


def compute_wait_rewards(signals: dict[str, RescoSignalState]) -> dict[str, float]:
    return {
        signal_id: -float(signal.observation.total_wait)
        for signal_id, signal in signals.items()
    }


def compute_pressure_rewards(signals: dict[str, RescoSignalState]) -> dict[str, float]:
    rewards: dict[str, float] = {}
    for signal_id, signal in signals.items():
        entering_queued = float(signal.observation.total_queued)
        exiting_queued = 0.0
        for lane_id in signal.outbound_lanes:
            dwn_signal_id = signal.out_lane_to_signal_id.get(lane_id)
            if dwn_signal_id in signals:
                exiting_queued += float(signals[dwn_signal_id].observation.get_lane(lane_id).queued)
        rewards[signal_id] = -(entering_queued - exiting_queued)
    return rewards
