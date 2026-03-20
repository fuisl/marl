from __future__ import annotations

from typing import Any

import pytest

import marl_env.traci_adapter as traci_adapter_mod
from marl_env.traci_adapter import TraCIAdapter


class _FakeSimulationAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def getTime(self) -> float:
        self.calls.append(("getTime", ()))
        return 12.5

    def getMinExpectedNumber(self) -> int:
        self.calls.append(("getMinExpectedNumber", ()))
        return 7

    def getDepartedNumber(self) -> int:
        self.calls.append(("getDepartedNumber", ()))
        return 3

    def getDepartedIDList(self) -> list[str]:
        self.calls.append(("getDepartedIDList", ()))
        return ["veh_dep_1"]

    def getArrivedNumber(self) -> int:
        self.calls.append(("getArrivedNumber", ()))
        return 2

    def getArrivedIDList(self) -> list[str]:
        self.calls.append(("getArrivedIDList", ()))
        return ["veh_arr_1"]

    def getStartingTeleportNumber(self) -> int:
        self.calls.append(("getStartingTeleportNumber", ()))
        return 1


class _FakeTrafficLightAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def getIDList(self) -> list[str]:
        self.calls.append(("getIDList", ()))
        return ["J1", "J2"]

    def getPhase(self, tl_id: str) -> int:
        self.calls.append(("getPhase", (tl_id,)))
        return 4

    def setPhase(self, tl_id: str, phase_index: int) -> None:
        self.calls.append(("setPhase", (tl_id, phase_index)))

    def setPhaseDuration(self, tl_id: str, duration: float) -> None:
        self.calls.append(("setPhaseDuration", (tl_id, duration)))

    def getAllProgramLogics(self, tl_id: str) -> list[str]:
        self.calls.append(("getAllProgramLogics", (tl_id,)))
        return ["logic"]

    def getControlledLanes(self, tl_id: str) -> tuple[str, ...]:
        self.calls.append(("getControlledLanes", (tl_id,)))
        return ("lane_a", "lane_b")

    def getControlledLinks(self, tl_id: str) -> list[list[tuple[str, str, str]]]:
        self.calls.append(("getControlledLinks", (tl_id,)))
        return [[("lane_a", "lane_b", "via_0")]]

    def getRedYellowGreenState(self, tl_id: str) -> str:
        self.calls.append(("getRedYellowGreenState", (tl_id,)))
        return "GrGr"


class _FakeLaneAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def getLastStepVehicleNumber(self, lane_id: str) -> int:
        self.calls.append(("getLastStepVehicleNumber", (lane_id,)))
        return 9

    def getLastStepHaltingNumber(self, lane_id: str) -> int:
        self.calls.append(("getLastStepHaltingNumber", (lane_id,)))
        return 5

    def getWaitingTime(self, lane_id: str) -> float:
        self.calls.append(("getWaitingTime", (lane_id,)))
        return 11.0

    def getLastStepMeanSpeed(self, lane_id: str) -> float:
        self.calls.append(("getLastStepMeanSpeed", (lane_id,)))
        return 8.5

    def getLastStepOccupancy(self, lane_id: str) -> float:
        self.calls.append(("getLastStepOccupancy", (lane_id,)))
        return 0.35

    def getLength(self, lane_id: str) -> float:
        self.calls.append(("getLength", (lane_id,)))
        return 100.0

    def getMaxSpeed(self, lane_id: str) -> float:
        self.calls.append(("getMaxSpeed", (lane_id,)))
        return 13.89


class _FakeTraCIException(Exception):
    pass


class _FakeVehicleAPI:
    def __init__(self) -> None:
        self.subscribe_calls: list[tuple[str, list[int]]] = []
        self.subscription_results: dict[str, dict[int, float]] = {}
        self.raise_on_timeloss_subscribe = False
        self.get_timeloss_calls: list[str] = []

    def subscribe(self, veh_id: str, variables: list[int]) -> None:
        if self.raise_on_timeloss_subscribe and 0x8C in variables:
            raise _FakeTraCIException("unsupported variable")
        self.subscribe_calls.append((veh_id, variables))

    def getSubscriptionResults(self, veh_id: str) -> dict[int, float]:
        return self.subscription_results.get(veh_id, {})

    def getTimeLoss(self, veh_id: str) -> float:
        self.get_timeloss_calls.append(veh_id)
        return 4.0


class _FakeConnection:
    def __init__(self) -> None:
        self.simulation = _FakeSimulationAPI()
        self.trafficlight = _FakeTrafficLightAPI()
        self.lane = _FakeLaneAPI()
        self.vehicle = _FakeVehicleAPI()
        self.TraCIException = _FakeTraCIException
        self.simulation_step_calls = 0
        self.close_calls: list[bool] = []

    def simulationStep(self) -> None:
        self.simulation_step_calls += 1

    def close(self, wait: bool = False) -> None:
        self.close_calls.append(wait)


class _FakeLibsumoModule:
    def __init__(self) -> None:
        self.start_calls: list[list[str]] = []
        self.close_calls: list[bool] = []
        self.TraCIException = _FakeTraCIException
        self.vehicle = _FakeVehicleAPI()

    def start(self, cmd: list[str]) -> None:
        self.start_calls.append(cmd)

    def close(self, wait: bool = False) -> None:
        self.close_calls.append(wait)

    def getVersion(self) -> tuple[int, str]:
        return (22, "SUMO 1.26.0")


def test_start_and_close_libsumo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_libsumo = _FakeLibsumoModule()
    monkeypatch.setattr(traci_adapter_mod, "traci", fake_libsumo)

    adapter = TraCIAdapter("net.net.xml", "route.rou.xml")
    adapter.start()

    assert fake_libsumo.start_calls == [adapter._sumo_cmd]
    assert adapter._conn is fake_libsumo

    adapter.close(wait=False)
    assert fake_libsumo.close_calls == [False]
    assert adapter._conn is None


def test_subscribe_vehicle_falls_back_when_timeloss_unsupported() -> None:
    adapter = TraCIAdapter(
        "net.net.xml",
        "route.rou.xml",
        timeloss_subscription_policy="fallback",
    )
    conn = _FakeConnection()
    conn.vehicle.raise_on_timeloss_subscribe = True
    adapter._conn = conn

    adapter.subscribe_vehicle("veh-1")
    adapter.subscribe_vehicle("veh-2")

    assert conn.vehicle.subscribe_calls == [
        ("veh-1", [0x87]),
        ("veh-2", [0x87]),
    ]


def test_vehicle_metrics_use_subscription_or_direct_fallback() -> None:
    adapter = TraCIAdapter(
        "net.net.xml",
        "route.rou.xml",
        timeloss_subscription_policy="fallback",
    )
    conn = _FakeConnection()
    adapter._conn = conn

    conn.vehicle.subscription_results["veh-sub"] = {0x87: 2.0, 0x8C: 3.0}
    assert adapter.get_vehicle_benchmark_metrics("veh-sub") == (2.0, 3.0)

    conn.vehicle.subscription_results["veh-direct"] = {0x87: 1.5}
    assert adapter.get_vehicle_benchmark_metrics("veh-direct") == (1.5, 4.0)

    adapter._vehicle_timeloss_sub_supported = False
    conn.vehicle.subscription_results["veh-compat"] = {0x87: 6.0}
    assert adapter.get_vehicle_benchmark_metrics("veh-compat") == (6.0, 6.0)


def test_subscribe_vehicle_strict_policy_raises() -> None:
    adapter = TraCIAdapter(
        "net.net.xml",
        "route.rou.xml",
        timeloss_subscription_policy="strict",
    )
    conn = _FakeConnection()
    conn.vehicle.raise_on_timeloss_subscribe = True
    adapter._conn = conn

    with pytest.raises(_FakeTraCIException):
        adapter.subscribe_vehicle("veh-1")


def test_invalid_timeloss_policy_raises() -> None:
    with pytest.raises(ValueError, match="timeloss_subscription_policy"):
        TraCIAdapter("net.net.xml", "route.rou.xml", timeloss_subscription_policy="invalid")


def test_all_interface_methods_forward_to_connection() -> None:
    adapter = TraCIAdapter("net.net.xml", "route.rou.xml")
    conn = _FakeConnection()
    adapter._conn = conn

    adapter.simulation_step()
    assert conn.simulation_step_calls == 1

    assert adapter.current_time == 12.5
    assert adapter.min_expected_vehicles == 7

    assert adapter.get_traffic_light_ids() == ["J1", "J2"]
    assert adapter.get_phase("J1") == 4
    adapter.set_phase("J1", 2)
    adapter.set_phase_duration("J1", 9.0)
    assert adapter.get_program_logic("J1") == ["logic"]
    assert adapter.get_controlled_lanes("J1") == ["lane_a", "lane_b"]
    assert adapter.get_controlled_links("J1") == [[("lane_a", "lane_b", "via_0")]]
    assert adapter.get_red_yellow_green_state("J1") == "GrGr"

    assert adapter.get_lane_vehicle_count("lane_a") == 9
    assert adapter.get_lane_halting_number("lane_a") == 5
    assert adapter.get_lane_waiting_time("lane_a") == 11.0
    assert adapter.get_lane_mean_speed("lane_a") == 8.5
    assert adapter.get_lane_occupancy("lane_a") == 0.35
    assert adapter.get_lane_length("lane_a") == 100.0
    assert adapter.get_lane_max_speed("lane_a") == 13.89

    assert adapter.get_departed_number() == 3
    assert adapter.get_departed_ids() == ["veh_dep_1"]
    assert adapter.get_arrived_number() == 2
    assert adapter.get_arrived_ids() == ["veh_arr_1"]
    assert adapter.get_teleported_number() == 1

    adapter.close(wait=False)
    assert conn.close_calls == [False]


def test_methods_raise_without_connection() -> None:
    adapter = TraCIAdapter("net.net.xml", "route.rou.xml")
    with pytest.raises(RuntimeError, match="No active SUMO connection"):
        adapter.simulation_step()
