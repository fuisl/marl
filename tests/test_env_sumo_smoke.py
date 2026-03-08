from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from marl_env.sumo_env import TrafficSignalEnv


def _build_net_from_nodes_edges(
    workdir: Path,
    net_name: str,
    nodes_xml: str,
    edges_xml: str,
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    nod_file = workdir / f"{net_name}.nod.xml"
    edg_file = workdir / f"{net_name}.edg.xml"
    net_file = workdir / f"{net_name}.net.xml"

    nod_file.write_text(nodes_xml.strip() + "\n", encoding="utf-8")
    edg_file.write_text(edges_xml.strip() + "\n", encoding="utf-8")

    cmd = [
        "netconvert",
        f"--node-files={nod_file}",
        f"--edge-files={edg_file}",
        f"--output-file={net_file}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return net_file


def _build_single_intersection_net(workdir: Path) -> Path:
    nodes_xml = """
<nodes>
    <node id="n" x="0" y="100" type="priority"/>
    <node id="s" x="0" y="-100" type="priority"/>
    <node id="e" x="100" y="0" type="priority"/>
    <node id="w" x="-100" y="0" type="priority"/>
    <node id="J1" x="0" y="0" type="traffic_light"/>
</nodes>
"""
    edges_xml = """
<edges>
    <edge id="n_J1" from="n" to="J1" numLanes="1" speed="13.89"/>
    <edge id="J1_n" from="J1" to="n" numLanes="1" speed="13.89"/>
    <edge id="s_J1" from="s" to="J1" numLanes="1" speed="13.89"/>
    <edge id="J1_s" from="J1" to="s" numLanes="1" speed="13.89"/>
    <edge id="e_J1" from="e" to="J1" numLanes="1" speed="13.89"/>
    <edge id="J1_e" from="J1" to="e" numLanes="1" speed="13.89"/>
    <edge id="w_J1" from="w" to="J1" numLanes="1" speed="13.89"/>
    <edge id="J1_w" from="J1" to="w" numLanes="1" speed="13.89"/>
</edges>
"""
    return _build_net_from_nodes_edges(workdir, "single", nodes_xml, edges_xml)


def _build_two_intersection_net(workdir: Path) -> Path:
    nodes_xml = """
<nodes>
    <node id="w" x="-220" y="0" type="priority"/>
    <node id="e" x="220" y="0" type="priority"/>
    <node id="n1" x="-100" y="120" type="priority"/>
    <node id="s1" x="-100" y="-120" type="priority"/>
    <node id="n2" x="100" y="120" type="priority"/>
    <node id="s2" x="100" y="-120" type="priority"/>
    <node id="J1" x="-100" y="0" type="traffic_light"/>
    <node id="J2" x="100" y="0" type="traffic_light"/>
</nodes>
"""
    edges_xml = """
<edges>
    <edge id="w_J1" from="w" to="J1" numLanes="1" speed="13.89"/>
    <edge id="J1_w" from="J1" to="w" numLanes="1" speed="13.89"/>
    <edge id="n1_J1" from="n1" to="J1" numLanes="1" speed="13.89"/>
    <edge id="J1_n1" from="J1" to="n1" numLanes="1" speed="13.89"/>
    <edge id="s1_J1" from="s1" to="J1" numLanes="1" speed="13.89"/>
    <edge id="J1_s1" from="J1" to="s1" numLanes="1" speed="13.89"/>

    <edge id="J1_J2" from="J1" to="J2" numLanes="1" speed="13.89"/>
    <edge id="J2_J1" from="J2" to="J1" numLanes="1" speed="13.89"/>

    <edge id="n2_J2" from="n2" to="J2" numLanes="1" speed="13.89"/>
    <edge id="J2_n2" from="J2" to="n2" numLanes="1" speed="13.89"/>
    <edge id="s2_J2" from="s2" to="J2" numLanes="1" speed="13.89"/>
    <edge id="J2_s2" from="J2" to="s2" numLanes="1" speed="13.89"/>
    <edge id="J2_e" from="J2" to="e" numLanes="1" speed="13.89"/>
    <edge id="e_J2" from="e" to="J2" numLanes="1" speed="13.89"/>
</edges>
"""
    return _build_net_from_nodes_edges(workdir, "double", nodes_xml, edges_xml)


def _count_traffic_lights(net_file: Path) -> int:
    root = ET.parse(net_file).getroot()
    return len({node.attrib["id"] for node in root.iter("tlLogic")})


def _collect_route_paths(
    net_file: Path,
    sumolib: object,
    max_paths: int = 8,
) -> list[str]:
    net = sumolib.net.readNet(str(net_file), withInternal=False)
    edges = [edge for edge in net.getEdges() if not edge.getID().startswith(":")]
    route_paths: list[str] = []

    for from_edge in edges:
        for to_edge in edges:
            if from_edge.getID() == to_edge.getID():
                continue
            path, _ = net.getShortestPath(from_edge, to_edge)
            if not path or len(path) < 2:
                continue
            edge_string = " ".join(edge.getID() for edge in path)
            if edge_string in route_paths:
                continue
            route_paths.append(edge_string)
            if len(route_paths) >= max_paths:
                return route_paths

    return route_paths


def _write_route_file(
    net_file: Path,
    route_file: Path,
    sumolib: object,
    n_vehicles: int,
) -> None:
    route_paths = _collect_route_paths(net_file, sumolib=sumolib)
    if not route_paths:
        raise RuntimeError(f"Could not build any valid route path for {net_file}.")

    lines = [
        "<routes>",
        '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9"/>',
    ]
    for i, route_edges in enumerate(route_paths):
        lines.append(f'  <route id="r{i}" edges="{route_edges}"/>')

    for i in range(n_vehicles):
        route_id = f"r{i % len(route_paths)}"
        lines.append(
            f'  <vehicle id="veh{i}" type="car" route="{route_id}" depart="{float(i):.1f}"/>'
        )
    lines.append("</routes>")
    route_file.write_text("\n".join(lines), encoding="utf-8")


def _sample_valid_actions(mask: torch.Tensor) -> torch.Tensor:
    actions: list[int] = []
    for row in mask:
        valid = torch.where(row)[0]
        assert valid.numel() > 0
        idx = int(torch.randint(valid.numel(), (1,)).item())
        actions.append(int(valid[idx].item()))
    return torch.tensor(actions, dtype=torch.long)


@pytest.fixture(scope="session")
def tiny_sumo_networks(
    tmp_path_factory: pytest.TempPathFactory,
    sumo_stack: dict[str, object],
) -> dict[str, tuple[Path, Path]]:
    sumolib = sumo_stack["sumolib"]
    root = tmp_path_factory.mktemp("sumo_smoke")

    single_net = _build_single_intersection_net(root / "single")
    double_net = _build_two_intersection_net(root / "double")
    if _count_traffic_lights(single_net) != 1:
        pytest.skip(f"Expected 1 traffic light in {single_net}.")
    if _count_traffic_lights(double_net) != 2:
        pytest.skip(f"Expected 2 traffic lights in {double_net}.")

    single_route = root / "single.rou.xml"
    double_route = root / "double.rou.xml"
    rollout_route = root / "rollout.rou.xml"
    _write_route_file(single_net, single_route, sumolib=sumolib, n_vehicles=80)
    _write_route_file(double_net, double_route, sumolib=sumolib, n_vehicles=120)
    _write_route_file(double_net, rollout_route, sumolib=sumolib, n_vehicles=360)

    return {
        "single": (single_net, single_route),
        "double": (double_net, double_route),
        "rollout": (double_net, rollout_route),
    }


@pytest.mark.sumo
def test_sumo_smoke_single_intersection(
    tiny_sumo_networks: dict[str, tuple[Path, Path]],
) -> None:
    net_file, route_file = tiny_sumo_networks["single"]
    env = TrafficSignalEnv(
        net_file=str(net_file),
        route_file=str(route_file),
        delta_t=1,
        reward_mode="combined",
        sumo_binary="sumo",
        gui=False,
        begin_time=0,
        end_time=600,
        min_green_duration=0,
        yellow_duration=2,
        all_red_duration=1,
    )

    try:
        td = env.reset()
        assert env.n_agents == 1

        obs_shape = td["agents", "observation"].shape
        mask_shape = td["agents", "action_mask"].shape
        tl_id = env.tl_ids[0]
        phase_history = [env.adapter.get_phase(tl_id)]

        for step in range(10):
            mask = td["agents", "action_mask"]
            assert mask.shape == mask_shape
            assert torch.all(mask.any(dim=1))

            if env.num_actions >= 2:
                action = torch.tensor([step % 2], dtype=torch.long)
            else:
                action = _sample_valid_actions(mask)

            td = env.step(action)
            phase_history.append(env.adapter.get_phase(tl_id))

            assert td["agents", "observation"].shape == obs_shape
            assert td["agents", "action_mask"].shape == mask_shape
            assert torch.isfinite(td["agents", "observation"]).all()
            assert torch.isfinite(td["agents", "reward"]).all()

        if env.num_actions >= 2:
            assert len(set(phase_history)) >= 2
    finally:
        env.close()


@pytest.mark.sumo
def test_sumo_smoke_two_intersections_graph_and_stepping(
    tiny_sumo_networks: dict[str, tuple[Path, Path]],
) -> None:
    net_file, route_file = tiny_sumo_networks["double"]
    env = TrafficSignalEnv(
        net_file=str(net_file),
        route_file=str(route_file),
        delta_t=1,
        reward_mode="combined",
        sumo_binary="sumo",
        gui=False,
        begin_time=0,
        end_time=600,
        min_green_duration=0,
    )

    try:
        td = env.reset()
        assert env.n_agents == 2
        assert td["edge_index"].shape[0] == 2
        assert td["edge_index"].shape[1] > 0

        obs_shape = td["agents", "observation"].shape
        mask_shape = td["agents", "action_mask"].shape

        for _ in range(20):
            mask = td["agents", "action_mask"]
            assert mask.shape == mask_shape
            assert torch.all(mask.any(dim=1))

            td = env.step(_sample_valid_actions(mask))
            assert td["agents", "observation"].shape == obs_shape
            assert td["agents", "action_mask"].shape == mask_shape
            assert torch.isfinite(td["agents", "observation"]).all()
            assert torch.isfinite(td["agents", "reward"]).all()

            if td["done"].item():
                break
    finally:
        env.close()


@pytest.mark.sumo
def test_sumo_forced_switching_sequence_and_elapsed_reset(
    tiny_sumo_networks: dict[str, tuple[Path, Path]],
) -> None:
    net_file, route_file = tiny_sumo_networks["single"]
    env = TrafficSignalEnv(
        net_file=str(net_file),
        route_file=str(route_file),
        delta_t=1,
        reward_mode="combined",
        sumo_binary="sumo",
        gui=False,
        begin_time=0,
        end_time=600,
        min_green_duration=0,
        yellow_duration=2,
        all_red_duration=1,
    )

    try:
        env.reset()
        if env.num_actions < 2:
            pytest.skip("Network has fewer than 2 controllable actions.")

        tl_id = env.tl_ids[0]
        prev_green = env._current_green[tl_id]
        saw_green_change = False
        saw_elapsed_reset = False
        saw_non_green_phase = False

        for step in range(12):
            env.step(torch.tensor([step % 2], dtype=torch.long))
            applied_phase = env.adapter.get_phase(tl_id)
            if applied_phase not in env._green_phases[tl_id]:
                saw_non_green_phase = True

            current_green = env._current_green[tl_id]
            if current_green != prev_green:
                saw_green_change = True
                if env._elapsed_green[tl_id] == 0.0:
                    saw_elapsed_reset = True
            prev_green = current_green

        assert env._elapsed_green[tl_id] >= 0.0
        assert saw_green_change
        assert saw_elapsed_reset

        state = env.constraints._agent_state[tl_id]
        if state.yellow_phase_map or state.all_red_phase_map:
            assert saw_non_green_phase
    finally:
        env.close()


@pytest.mark.sumo
def test_random_rollout_sanity_100_steps(
    tiny_sumo_networks: dict[str, tuple[Path, Path]],
) -> None:
    net_file, route_file = tiny_sumo_networks["rollout"]
    env = TrafficSignalEnv(
        net_file=str(net_file),
        route_file=str(route_file),
        delta_t=1,
        reward_mode="combined",
        sumo_binary="sumo",
        gui=False,
        begin_time=0,
        end_time=1200,
        min_green_duration=0,
    )

    try:
        td = env.reset()
        obs_shape = td["agents", "observation"].shape
        mask_shape = td["agents", "action_mask"].shape
        edge_index = td["edge_index"].clone()

        done = bool(td.get("done", torch.tensor([False])).item())
        steps = 0

        for _ in range(300):
            mask = td["agents", "action_mask"]
            assert mask.shape == mask_shape
            assert torch.all(mask.any(dim=1))

            actions = _sample_valid_actions(mask)
            selected_validity = torch.gather(mask, 1, actions.unsqueeze(-1))
            assert selected_validity.all()

            td = env.step(actions)
            steps += 1

            assert td["agents", "observation"].shape == obs_shape
            assert td["agents", "action_mask"].shape == mask_shape
            assert torch.equal(td["edge_index"], edge_index)
            assert torch.isfinite(td["agents", "observation"]).all()
            assert torch.isfinite(td["agents", "reward"]).all()

            done = bool(td["done"].item())
            if done and steps >= 100:
                break

        assert steps >= 100
        # assert done
    finally:
        env.close()
