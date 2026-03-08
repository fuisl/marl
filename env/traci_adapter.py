"""Thin adapter over TraCI / libsumo.

All SUMO interaction goes through this module so that swapping TraCI for
libsumo later is a one-line change.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Try libsumo first (faster, same API); fall back to traci.
if "LIBSUMO_AS_TRACI" in os.environ:
    import libsumo as traci  # type: ignore[import-untyped]
else:
    import traci  # type: ignore[import-untyped]


class TraCIAdapter:
    """Manages SUMO process lifecycle and provides typed accessors."""

    def __init__(
        self,
        net_file: str,
        route_file: str,
        *,
        sumo_binary: str = "sumo",
        additional_files: list[str] | None = None,
        gui: bool = False,
        delta_t: int = 5,
        begin_time: int = 0,
        end_time: int = 3600,
        extra_args: list[str] | None = None,
        label: str = "default",
    ) -> None:
        self.net_file = net_file
        self.route_file = route_file
        self.delta_t = delta_t
        self.begin_time = begin_time
        self.end_time = end_time
        self.label = label

        binary = "sumo-gui" if gui else sumo_binary
        self._sumo_cmd: list[str] = [
            binary,
            "-n", net_file,
            "-r", route_file,
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "-b", str(begin_time),
            "-e", str(end_time),
            "--no-step-log", "True",
            "--no-warnings", "True",
        ]
        if additional_files:
            self._sumo_cmd += ["-a", ",".join(additional_files)]
        if extra_args:
            self._sumo_cmd += extra_args

        self._conn: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        traci.start(self._sumo_cmd, label=self.label)
        self._conn = traci

    def close(self) -> None:
        if self._conn is not None:
            traci.close()
            self._conn = None

    def simulation_step(self) -> None:
        """Advance simulation by one SUMO step (typically 1 s)."""
        traci.simulationStep()

    @property
    def current_time(self) -> float:
        return traci.simulation.getTime()

    @property
    def min_expected_vehicles(self) -> int:
        return traci.simulation.getMinExpectedNumber()

    # ------------------------------------------------------------------
    # Traffic-light accessors
    # ------------------------------------------------------------------
    def get_traffic_light_ids(self) -> list[str]:
        return list(traci.trafficlight.getIDList())

    def get_phase(self, tl_id: str) -> int:
        return traci.trafficlight.getPhase(tl_id)

    def set_phase(self, tl_id: str, phase_index: int) -> None:
        traci.trafficlight.setPhase(tl_id, phase_index)

    def set_phase_duration(self, tl_id: str, duration: float) -> None:
        traci.trafficlight.setPhaseDuration(tl_id, duration)

    def get_program_logic(self, tl_id: str) -> Any:
        return traci.trafficlight.getAllProgramLogics(tl_id)

    def get_controlled_lanes(self, tl_id: str) -> list[str]:
        return list(traci.trafficlight.getControlledLanes(tl_id))

    def get_controlled_links(self, tl_id: str) -> list[list[tuple[str, str, str]]]:
        return traci.trafficlight.getControlledLinks(tl_id)

    def get_red_yellow_green_state(self, tl_id: str) -> str:
        return traci.trafficlight.getRedYellowGreenState(tl_id)

    # ------------------------------------------------------------------
    # Lane / detector accessors
    # ------------------------------------------------------------------
    def get_lane_vehicle_count(self, lane_id: str) -> int:
        return traci.lane.getLastStepVehicleNumber(lane_id)

    def get_lane_halting_number(self, lane_id: str) -> int:
        return traci.lane.getLastStepHaltingNumber(lane_id)

    def get_lane_waiting_time(self, lane_id: str) -> float:
        return traci.lane.getWaitingTime(lane_id)

    def get_lane_mean_speed(self, lane_id: str) -> float:
        return traci.lane.getLastStepMeanSpeed(lane_id)

    def get_lane_occupancy(self, lane_id: str) -> float:
        return traci.lane.getLastStepOccupancy(lane_id)

    def get_lane_length(self, lane_id: str) -> float:
        return traci.lane.getLength(lane_id)

    def get_lane_max_speed(self, lane_id: str) -> float:
        return traci.lane.getMaxSpeed(lane_id)

    # ------------------------------------------------------------------
    # Simulation-wide stats
    # ------------------------------------------------------------------
    def get_departed_number(self) -> int:
        return traci.simulation.getDepartedNumber()

    def get_arrived_number(self) -> int:
        return traci.simulation.getArrivedNumber()

    def get_teleported_number(self) -> int:
        return traci.simulation.getStartingTeleportNumber()
