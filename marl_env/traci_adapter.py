"""Thin adapter over TraCI / libsumo.

All SUMO interaction goes through this module so that swapping TraCI for
libsumo later is a one-line change.
"""

from __future__ import annotations

import os
from typing import Any

# libsumo is API-compatible with traci but runs in-process.
_USING_LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

if _USING_LIBSUMO:
    import libsumo as traci  # type: ignore[import-untyped]
else:
    import traci  # type: ignore[import-untyped]


class TraCIAdapter:
    """Manages SUMO process lifecycle and provides typed accessors.

    For plain traci, methods use a per-label connection object so that
    multiple environments can coexist in the same process.

    For libsumo, there is no separate socket connection object; the module
    itself is the API surface.
    """

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
        self._using_libsumo = _USING_LIBSUMO

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
        if self._conn is not None:
            raise RuntimeError("TraCIAdapter.start() called while a connection is already open.")

        if self._using_libsumo:
            # libsumo uses the module itself as the API surface.
            traci.start(self._sumo_cmd)
            self._conn = traci
        else:
            # traci supports labeled connections.
            traci.start(self._sumo_cmd, label=self.label)
            self._conn = traci.getConnection(self.label)  # type: ignore[assignment]

    def close(self, wait: bool = False) -> None:
        if self._conn is None:
            return

        try:
            # libsumo's close() takes no arguments; traci's accepts close(False).
            try:
                self._conn.close(wait)
            except TypeError:
                self._conn.close()
        finally:
            self._conn = None

    def simulation_step(self) -> None:
        """Advance simulation by one SUMO step."""
        self._require_conn().simulationStep()

    @property
    def current_time(self) -> float:
        return self._require_conn().simulation.getTime()

    @property
    def min_expected_vehicles(self) -> int:
        return self._require_conn().simulation.getMinExpectedNumber()

    # ------------------------------------------------------------------
    # Traffic-light accessors
    # ------------------------------------------------------------------
    def get_traffic_light_ids(self) -> list[str]:
        return list(self._require_conn().trafficlight.getIDList())

    def get_phase(self, tl_id: str) -> int:
        return self._require_conn().trafficlight.getPhase(tl_id)

    def set_phase(self, tl_id: str, phase_index: int) -> None:
        self._require_conn().trafficlight.setPhase(tl_id, phase_index)

    def set_phase_duration(self, tl_id: str, duration: float) -> None:
        self._require_conn().trafficlight.setPhaseDuration(tl_id, duration)

    def get_program_logic(self, tl_id: str) -> Any:
        return self._require_conn().trafficlight.getAllProgramLogics(tl_id)

    def get_controlled_lanes(self, tl_id: str) -> list[str]:
        return list(self._require_conn().trafficlight.getControlledLanes(tl_id))

    def get_controlled_links(self, tl_id: str) -> list[list[tuple[str, str, str]]]:
        return list(self._require_conn().trafficlight.getControlledLinks(tl_id))

    def get_red_yellow_green_state(self, tl_id: str) -> str:
        return self._require_conn().trafficlight.getRedYellowGreenState(tl_id)

    # ------------------------------------------------------------------
    # Lane accessors
    # ------------------------------------------------------------------
    def get_lane_vehicle_count(self, lane_id: str) -> int:
        return self._require_conn().lane.getLastStepVehicleNumber(lane_id)

    def get_lane_halting_number(self, lane_id: str) -> int:
        return self._require_conn().lane.getLastStepHaltingNumber(lane_id)

    def get_lane_waiting_time(self, lane_id: str) -> float:
        return self._require_conn().lane.getWaitingTime(lane_id)

    def get_lane_mean_speed(self, lane_id: str) -> float:
        return self._require_conn().lane.getLastStepMeanSpeed(lane_id)

    def get_lane_occupancy(self, lane_id: str) -> float:
        return self._require_conn().lane.getLastStepOccupancy(lane_id)

    def get_lane_length(self, lane_id: str) -> float:
        return self._require_conn().lane.getLength(lane_id)

    def get_lane_max_speed(self, lane_id: str) -> float:
        return self._require_conn().lane.getMaxSpeed(lane_id)

    # ------------------------------------------------------------------
    # Simulation-wide stats
    # ------------------------------------------------------------------
    def get_departed_number(self) -> int:
        return self._require_conn().simulation.getDepartedNumber()

    def get_departed_ids(self) -> list[str]:
        return list(self._require_conn().simulation.getDepartedIDList())

    def get_arrived_number(self) -> int:
        return self._require_conn().simulation.getArrivedNumber()

    def get_arrived_ids(self) -> list[str]:
        return list(self._require_conn().simulation.getArrivedIDList())

    def get_teleported_number(self) -> int:
        return self._require_conn().simulation.getStartingTeleportNumber()

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _require_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("No active SUMO connection. Call start() first.")
        return self._conn
