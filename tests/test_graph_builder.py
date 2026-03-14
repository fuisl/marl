from __future__ import annotations

from dataclasses import dataclass

import torch

import marl_env.graph_builder as graph_builder_mod
from marl_env.graph_builder import GraphBuilder


class FakeEdge:
    def __init__(self, from_node: "FakeNode", to_node: "FakeNode", length: float, lanes: int) -> None:
        self._from_node = from_node
        self._to_node = to_node
        self._length = length
        self._lanes = lanes

    def getFromNode(self) -> "FakeNode":
        return self._from_node

    def getToNode(self) -> "FakeNode":
        return self._to_node

    def getLength(self) -> float:
        return self._length

    def getLaneNumber(self) -> int:
        return self._lanes


class FakeNode:
    def __init__(self, node_id: str, *, coord: tuple[float, float] = (0.0, 0.0)) -> None:
        self._id = node_id
        self._coord = coord
        self._outgoing: list[FakeEdge] = []
        self._incoming: list[FakeEdge] = []

    def getID(self) -> str:
        return self._id

    def getOutgoing(self) -> list[FakeEdge]:
        return self._outgoing

    def getIncoming(self) -> list[FakeEdge]:
        return self._incoming

    def getCoord(self) -> tuple[float, float]:
        return self._coord

    def add_outgoing(self, edge: FakeEdge) -> None:
        self._outgoing.append(edge)

    def add_incoming(self, edge: FakeEdge) -> None:
        self._incoming.append(edge)


class FakeLane:
    def __init__(self, edge: FakeEdge) -> None:
        self._edge = edge

    def getEdge(self) -> FakeEdge:
        return self._edge


class FakeTLS:
    def __init__(self, tls_id: str, connections: list[list[object]]) -> None:
        self._id = tls_id
        self._connections = connections

    def getID(self) -> str:
        return self._id

    def getConnections(self) -> list[list[object]]:
        return self._connections


class FakeNet:
    def __init__(self, nodes: dict[str, FakeNode], tls_list: list[FakeTLS]) -> None:
        self._nodes = nodes
        self._tls_list = tls_list

    def getNode(self, node_id: str) -> FakeNode:
        return self._nodes[node_id]

    def getTrafficLights(self) -> list[FakeTLS]:
        return self._tls_list


class FakeNetNoTLSApi:
    def __init__(self, nodes: dict[str, FakeNode]) -> None:
        self._nodes = nodes

    def getNode(self, node_id: str) -> FakeNode:
        return self._nodes[node_id]


@dataclass
class _FakeNetModule:
    fake_net: FakeNet

    def readNet(self, net_file: str, withInternal: bool = False) -> FakeNet:  # noqa: N803
        _ = net_file
        _ = withInternal
        return self.fake_net


@dataclass
class _FakeSumoLib:
    net: _FakeNetModule


def _connect(a: FakeNode, b: FakeNode, length: float, lanes: int) -> FakeEdge:
    edge = FakeEdge(a, b, length, lanes)
    a.add_outgoing(edge)
    b.add_incoming(edge)
    return edge


def _build_fake_graph(*, berlin_style_tls_ids: bool) -> tuple[object, list[str]]:
    n_a = FakeNode("A", coord=(0.0, 0.0))
    n_b = FakeNode("B", coord=(10.0, 5.0))
    n_c = FakeNode("C", coord=(20.0, 10.0))

    e_ab = _connect(n_a, n_b, 10.0, 2)
    e_ba = _connect(n_b, n_a, 11.0, 1)
    e_bc = _connect(n_b, n_c, 20.0, 3)
    e_cb = _connect(n_c, n_b, 21.0, 1)

    nodes = {"A": n_a, "B": n_b, "C": n_c}

    if berlin_style_tls_ids:
        tls_ids = ["GS_A", "GS_B", "GS_C"]
        tls_list = [
            # Each conn is [from_lane, to_lane, link_index]
            FakeTLS("GS_A", [[FakeLane(e_ba), FakeLane(e_ab), 0]]),
            FakeTLS("GS_B", [[FakeLane(e_ab), FakeLane(e_bc), 0]]),
            FakeTLS("GS_C", [[FakeLane(e_bc), FakeLane(e_cb), 0]]),
        ]
        fake_net = FakeNet(nodes=nodes, tls_list=tls_list)
        return fake_net, tls_ids

    tls_ids = ["A", "B", "C"]
    fake_net = FakeNetNoTLSApi(nodes=nodes)
    return fake_net, tls_ids


def _patch_sumolib(monkeypatch: object, fake_net: object) -> None:
    fake_sumolib = _FakeSumoLib(net=_FakeNetModule(fake_net=fake_net))
    monkeypatch.setattr(graph_builder_mod, "sumolib", fake_sumolib)


def test_build_graph_from_berlin_style_tls_ids(monkeypatch: object) -> None:
    fake_net, tl_ids = _build_fake_graph(berlin_style_tls_ids=True)
    _patch_sumolib(monkeypatch, fake_net)

    builder = GraphBuilder(net_file="unused.net.xml", tl_ids=tl_ids)
    edge_index, edge_attr = builder.build()

    assert builder.num_nodes == 3
    assert edge_attr is not None

    # Distances use the shortest undirected road segment between controllers.
    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert torch.allclose(
        edge_attr,
        torch.tensor(
            [[10.0, 2.0], [10.0, 2.0], [20.0, 3.0], [20.0, 3.0]],
            dtype=torch.float32,
        ),
    )


def test_build_graph_falls_back_to_node_ids_without_tls_api(monkeypatch: object) -> None:
    fake_net, tl_ids = _build_fake_graph(berlin_style_tls_ids=False)
    _patch_sumolib(monkeypatch, fake_net)

    builder = GraphBuilder(net_file="unused.net.xml", tl_ids=tl_ids)
    edge_index, edge_attr = builder.build()

    assert builder.num_nodes == 3
    assert edge_attr is not None
    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert torch.allclose(
        edge_attr,
        torch.tensor(
            [[10.0, 2.0], [10.0, 2.0], [20.0, 3.0], [20.0, 3.0]],
            dtype=torch.float32,
        ),
    )


def test_node_positions_align_with_tls_ids(monkeypatch: object) -> None:
    fake_net, tl_ids = _build_fake_graph(berlin_style_tls_ids=False)
    _patch_sumolib(monkeypatch, fake_net)

    builder = GraphBuilder(net_file="unused.net.xml", tl_ids=tl_ids)

    assert torch.allclose(
        builder.node_positions,
        torch.tensor(
            [[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]],
            dtype=torch.float32,
        ),
    )


def test_node_positions_use_centroid_for_controller_tls(monkeypatch: object) -> None:
    n_a = FakeNode("A", coord=(0.0, 0.0))
    n_b = FakeNode("B", coord=(20.0, 10.0))
    n_c = FakeNode("C", coord=(40.0, 0.0))

    e_ab = _connect(n_a, n_b, 10.0, 1)
    e_cb = _connect(n_c, n_b, 12.0, 1)

    fake_net = FakeNet(
        nodes={"A": n_a, "B": n_b, "C": n_c},
        tls_list=[
            FakeTLS("GS_BC", [[FakeLane(e_ab), FakeLane(e_cb), 0]]),
        ],
    )
    _patch_sumolib(monkeypatch, fake_net)

    builder = GraphBuilder(net_file="unused.net.xml", tl_ids=["GS_BC"])

    assert torch.allclose(
        builder.node_positions,
        torch.tensor([[30.0, 5.0]], dtype=torch.float32),
    )


def test_build_graph_connects_tls_through_unsignalized_nodes(monkeypatch: object) -> None:
    n_a = FakeNode("A", coord=(0.0, 0.0))
    n_x = FakeNode("X", coord=(5.0, 0.0))
    n_b = FakeNode("B", coord=(10.0, 0.0))

    e_ax = _connect(n_a, n_x, 5.0, 3)
    e_xa = _connect(n_x, n_a, 7.0, 2)
    e_xb = _connect(n_x, n_b, 11.0, 1)
    e_bx = _connect(n_b, n_x, 13.0, 4)

    fake_net = FakeNet(
        nodes={"A": n_a, "X": n_x, "B": n_b},
        tls_list=[
            FakeTLS("GS_A", [[FakeLane(e_xa), FakeLane(e_ax), 0]]),
            FakeTLS("GS_B", [[FakeLane(e_xb), FakeLane(e_bx), 0]]),
        ],
    )
    _patch_sumolib(monkeypatch, fake_net)

    builder = GraphBuilder(net_file="unused.net.xml", tl_ids=["GS_A", "GS_B"])
    edge_index, edge_attr = builder.build()

    assert edge_attr is not None
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert torch.allclose(
        edge_attr,
        torch.tensor(
            [[16.0, 1.0], [16.0, 1.0]],
            dtype=torch.float32,
        ),
    )
