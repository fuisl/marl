"""Thin adapter over libsumo.

All SUMO interaction goes through this module.
Legacy socket TraCI mode is deprecated and no longer used.
"""

from __future__ import annotations

import re
from typing import Any
import warnings

try:
    # libsumo is API-compatible with traci but runs in-process.
    import libsumo as traci  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "libsumo is required. Install SUMO Python bindings in the active environment."
    ) from exc


# Stable TraCI/libsumo vehicle variable IDs used for per-vehicle episode stats.
_VAR_ACCUM_WAITING: int = 0x87  # vehicle.getAccumulatedWaitingTime() == tc.VAR_ACCUMULATED_WAITING_TIME
_VAR_TIME_LOSS: int = 0x8C      # vehicle.getTimeLoss()              == tc.VAR_TIMELOSS
_MIN_SUPPORTED_SUMO_VERSION: tuple[int, int, int] = (1, 24, 0)
_TIMELOSS_SUB_POLICIES: set[str] = {"strict", "fallback"}


class TraCIAdapter:
    """Manages SUMO process lifecycle and provides typed accessors.

    Uses libsumo only. There is no separate socket connection object;
    the module itself is the API surface.
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
        timeloss_subscription_policy: str = "strict",
    ) -> None:
        self.net_file = net_file
        self.route_file = route_file
        self.delta_t = delta_t
        self.begin_time = begin_time
        self.end_time = end_time
        self.label = label
        if timeloss_subscription_policy not in _TIMELOSS_SUB_POLICIES:
            raise ValueError(
                "timeloss_subscription_policy must be one of "
                f"{sorted(_TIMELOSS_SUB_POLICIES)}"
            )
        self.timeloss_subscription_policy = timeloss_subscription_policy
        self._vehicle_timeloss_sub_supported: bool | None = None

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
        self._sumo_version: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._conn is not None:
            raise RuntimeError("TraCIAdapter.start() called while a connection is already open.")

        # libsumo uses the module itself as the API surface.
        traci.start(self._sumo_cmd)
        self._conn = traci
        self._validate_runtime_version()

    def close(self, wait: bool = False) -> None:
        if self._conn is None:
            return

        try:
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
    # Vehicle subscriptions (per-vehicle episode stats)
    # ------------------------------------------------------------------
    def subscribe_vehicle(self, vid: str) -> None:
        """Subscribe to per-vehicle stats for *vid*.

        Must be called once after the vehicle departs.  Subscription data
        is guaranteed to remain readable for one step after the vehicle
        exits the network (both TraCI and libsumo preserve this).

        Some SUMO server builds reject ``VAR_TIMELOSS`` (0x8c) for
        ``vehicle.subscribe`` even though other vehicle vars are supported.
        In that case we fall back to subscribing only accumulated waiting
        time for the rest of the run.
        """
        conn = self._require_conn()
        traci_exception = getattr(conn, "TraCIException", Exception)
        if self.timeloss_subscription_policy == "fallback" and self._vehicle_timeloss_sub_supported is False:
            conn.vehicle.subscribe(vid, [_VAR_ACCUM_WAITING])
            return

        try:
            conn.vehicle.subscribe(vid, [_VAR_ACCUM_WAITING, _VAR_TIME_LOSS])
            self._vehicle_timeloss_sub_supported = True
        except traci_exception:
            if self.timeloss_subscription_policy == "strict":
                raise
            if self._vehicle_timeloss_sub_supported is None:
                self._vehicle_timeloss_sub_supported = False
                warnings.warn(
                    "vehicle VAR_TIMELOSS subscription unsupported; falling back to "
                    "VAR_ACCUMULATED_WAITING_TIME only",
                    RuntimeWarning,
                    stacklevel=2,
                )
            conn.vehicle.subscribe(vid, [_VAR_ACCUM_WAITING])

    def get_vehicle_benchmark_metrics(self, vid: str) -> tuple[float, float]:
        """Return ``(accum_wait_s, delay_s)`` for one vehicle.

        Delay is taken from ``VAR_TIMELOSS`` when available. On older SUMO
        servers where ``VAR_TIMELOSS`` is unsupported for vehicle subscriptions,
        a compatibility fallback is used: ``delay_s := accum_wait_s``.
        """
        sub = self.get_vehicle_subscription_results(vid)
        accum_wait = float(sub.get(_VAR_ACCUM_WAITING, 0.0))

        if _VAR_TIME_LOSS in sub:
            return accum_wait, float(sub[_VAR_TIME_LOSS])

        # Compatibility fallback for runtimes that reject vehicle VAR_TIMELOSS.
        if (
            self.timeloss_subscription_policy == "fallback"
            and self._vehicle_timeloss_sub_supported is False
        ):
            return accum_wait, accum_wait

        # Some runtimes do not return subscribed VAR_TIMELOSS but still support
        # direct lookup while the vehicle is still queryable.
        try:
            delay = float(self._require_conn().vehicle.getTimeLoss(vid))
            return accum_wait, delay
        except Exception:
            pass

        return accum_wait, 0.0

    def get_vehicle_subscription_results(self, vid: str) -> dict[int, Any]:
        """Return the last subscription snapshot for *vid*.

        Returns an empty dict if the vehicle was never subscribed or if
        results are unavailable (safe fallback).
        """
        return dict(self._require_conn().vehicle.getSubscriptionResults(vid) or {})

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _require_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("No active SUMO connection. Call start() first.")
        return self._conn

    def _validate_runtime_version(self) -> None:
        conn = self._require_conn()
        get_version = getattr(conn, "getVersion", None)
        if get_version is None:
            return

        try:
            _, version_text = get_version()
        except Exception:
            return

        self._sumo_version = str(version_text)
        parsed = self._parse_sumo_version(self._sumo_version)
        if parsed is None:
            return

        if parsed < _MIN_SUPPORTED_SUMO_VERSION:
            warnings.warn(
                "Detected SUMO "
                f"{self._sumo_version}. Recommended version is "
                f">={'.'.join(map(str, _MIN_SUPPORTED_SUMO_VERSION))} for stable libsumo behavior.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _parse_sumo_version(version_text: str) -> tuple[int, int, int] | None:
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_text)
        if match is None:
            return None
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
