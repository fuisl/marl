from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import marl_env.traci_adapter as traci_adapter_mod
from marl_env.traci_adapter import TraCIAdapter
from tests.test_env_sumo_smoke import _build_single_intersection_net, _write_route_file


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

    def getArrivedNumber(self) -> int:
        self.calls.append(("getArrivedNumber", ()))
        return 2

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


class _FakeConnection:
    def __init__(self) -> None:
        self.simulation = _FakeSimulationAPI()
        self.trafficlight = _FakeTrafficLightAPI()
        self.lane = _FakeLaneAPI()
        self.simulation_step_calls = 0
        self.close_calls: list[bool] = []

    def simulationStep(self) -> None:
        self.simulation_step_calls += 1

    def close(self, wait: bool = False) -> None:
        self.close_calls.append(wait)


class _FakeTraciModule:
    def __init__(self, conn: _FakeConnection) -> None:
        self.conn = conn
        self.start_calls: list[tuple[list[str], str | None]] = []
        self.get_connection_calls: list[str] = []

    def start(self, cmd: list[str], label: str | None = None) -> None:
        self.start_calls.append((cmd, label))

    def getConnection(self, label: str) -> _FakeConnection:
        self.get_connection_calls.append(label)
        return self.conn


class _FakeLibsumoModule:
    def __init__(self) -> None:
        self.start_calls: list[list[str]] = []
        self.close_calls: list[bool] = []

    def start(self, cmd: list[str]) -> None:
        self.start_calls.append(cmd)

    def close(self, wait: bool = False) -> None:
        self.close_calls.append(wait)


def test_start_and_close_plain_traci(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_conn = _FakeConnection()
    fake_traci = _FakeTraciModule(fake_conn)
    monkeypatch.setattr(traci_adapter_mod, "traci", fake_traci)

    adapter = TraCIAdapter("net.net.xml", "route.rou.xml", label="test-label")
    adapter._using_libsumo = False

    adapter.start()
    assert fake_traci.start_calls == [(adapter._sumo_cmd, "test-label")]
    assert fake_traci.get_connection_calls == ["test-label"]
    assert adapter._conn is fake_conn

    with pytest.raises(RuntimeError, match="already open"):
        adapter.start()

    adapter.close(wait=True)
    assert fake_conn.close_calls == [True]
    assert adapter._conn is None

    # Closing twice should be a no-op.
    adapter.close(wait=False)


def test_start_and_close_libsumo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_libsumo = _FakeLibsumoModule()
    monkeypatch.setattr(traci_adapter_mod, "traci", fake_libsumo)

    adapter = TraCIAdapter("net.net.xml", "route.rou.xml")
    adapter._using_libsumo = True
    adapter.start()

    assert fake_libsumo.start_calls == [adapter._sumo_cmd]
    assert adapter._conn is fake_libsumo

    adapter.close(wait=False)
    assert fake_libsumo.close_calls == [False]
    assert adapter._conn is None


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
    assert adapter.get_arrived_number() == 2
    assert adapter.get_teleported_number() == 1

    adapter.close(wait=False)
    assert conn.close_calls == [False]


def test_methods_raise_without_connection() -> None:
    adapter = TraCIAdapter("net.net.xml", "route.rou.xml")
    with pytest.raises(RuntimeError, match="No active SUMO connection"):
        adapter.simulation_step()


@pytest.mark.sumo
def test_traci_adapter_calls_real_sumo(
    tmp_path: Path,
    sumo_stack: dict[str, object],
) -> None:
    sumolib = sumo_stack["sumolib"]
    try:
        net_file = _build_single_intersection_net(tmp_path / "adapter_sumo")
        route_file = tmp_path / "adapter_sumo.rou.xml"
        _write_route_file(net_file, route_file, sumolib=sumolib, n_vehicles=40)
    except (subprocess.CalledProcessError, OSError) as exc:
        pytest.skip(f"SUMO tooling unavailable at runtime: {exc}")

    adapter = TraCIAdapter(
        net_file=str(net_file),
        route_file=str(route_file),
        begin_time=0,
        end_time=300,
        sumo_binary="sumo",
    )

    adapter.start()
    try:
        tl_ids = adapter.get_traffic_light_ids()
        assert tl_ids
        tl_id = tl_ids[0]

        current_phase = adapter.get_phase(tl_id)
        adapter.set_phase(tl_id, current_phase)
        adapter.set_phase_duration(tl_id, 3.0)
        assert adapter.get_program_logic(tl_id)
        assert isinstance(adapter.get_controlled_links(tl_id), list)
        assert isinstance(adapter.get_red_yellow_green_state(tl_id), str)

        lanes = adapter.get_controlled_lanes(tl_id)
        assert lanes
        lane_id = lanes[0]
        assert isinstance(adapter.get_lane_vehicle_count(lane_id), int)
        assert isinstance(adapter.get_lane_halting_number(lane_id), int)
        assert isinstance(adapter.get_lane_waiting_time(lane_id), float)
        assert isinstance(adapter.get_lane_mean_speed(lane_id), float)
        assert isinstance(adapter.get_lane_occupancy(lane_id), float)
        assert isinstance(adapter.get_lane_length(lane_id), float)
        assert isinstance(adapter.get_lane_max_speed(lane_id), float)

        adapter.simulation_step()
        assert adapter.current_time >= 0.0
        assert adapter.min_expected_vehicles >= 0
        assert adapter.get_departed_number() >= 0
        assert adapter.get_arrived_number() >= 0
        assert adapter.get_teleported_number() >= 0
    finally:
        adapter.close()
