"""Build the intersection adjacency graph from a SUMO network.

The graph is built once at environment init and stays fixed during training.
Node features change every step; topology does not.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

try:
    import sumolib  # type: ignore[import-untyped]
except ImportError:
    sumolib = None  # allow import without sumolib for typing


class GraphBuilder:
    """Builds a homogeneous intersection graph from a SUMO ``.net.xml``.

    Nodes  = controlled traffic-light intersections.
    Edges  = pairs of intersections connected by at least one road segment.

    Optional edge attributes include distance between intersections and
    the number of connecting lanes.
    """

    def __init__(self, net_file: str, tl_ids: list[str]) -> None:
        if sumolib is None:
            raise ImportError("sumolib is required — install SUMO tools.")

        self.net: Any = sumolib.net.readNet(net_file, withInternal=False)
        self.tl_ids = tl_ids
        self._id_to_idx = {tl_id: i for i, tl_id in enumerate(tl_ids)}
        self._tls_nodes_by_id = self._build_tls_node_map()
        self._node_to_tls_id = self._build_node_to_tls_map(self._tls_nodes_by_id)

        self._edge_index: Tensor | None = None
        self._edge_attr: Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> tuple[Tensor, Tensor | None]:
        """Return ``(edge_index, edge_attr)`` for the intersection graph.

        ``edge_index`` has shape ``[2, E]`` (COO format).
        ``edge_attr``  has shape ``[E, d_edge]`` or ``None``.
        """
        src_list: list[int] = []
        dst_list: list[int] = []
        attrs: list[list[float]] = []

        for tl_id in self.tl_ids:
            src_idx = self._id_to_idx[tl_id]
            neighbors = self._get_neighbor_tl_ids_for_tls(tl_id)

            for nbr_id, dist, n_lanes in neighbors:
                if nbr_id not in self._id_to_idx:
                    continue
                dst_idx = self._id_to_idx[nbr_id]
                # One direction per iteration; the reverse is added
                # when the neighbor node is processed as src_idx,
                # since _get_neighbor_tl_ids scans both incoming
                # and outgoing edges.
                src_list.append(src_idx)
                dst_list.append(dst_idx)
                attrs.append([dist, float(n_lanes)])

        if not src_list:
            self._edge_index = torch.zeros(2, 0, dtype=torch.long)
            self._edge_attr = None
        else:
            self._edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long
            )
            self._edge_attr = torch.tensor(attrs, dtype=torch.float32)

        return self._edge_index, self._edge_attr

    @property
    def edge_index(self) -> Tensor:
        if self._edge_index is None:
            self.build()
        assert self._edge_index is not None
        return self._edge_index

    @property
    def edge_attr(self) -> Tensor | None:
        if self._edge_index is None:
            self.build()
        return self._edge_attr

    @property
    def num_nodes(self) -> int:
        return len(self.tl_ids)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_neighbor_tl_ids(
        self, node: Any
    ) -> list[tuple[str, float, int]]:
        """Return ``(neighbor_tl_id, distance, n_lanes)`` for neighbours."""
        neighbors: list[tuple[str, float, int]] = []
        seen: set[str] = set()

        for edge in node.getOutgoing():
            to_node = edge.getToNode()
            nbr_id = to_node.getID()
            if nbr_id in seen or nbr_id == node.getID():
                continue
            if nbr_id not in self._id_to_idx:
                continue
            seen.add(nbr_id)

            dist = edge.getLength()
            n_lanes = edge.getLaneNumber()
            neighbors.append((nbr_id, dist, n_lanes))

        for edge in node.getIncoming():
            from_node = edge.getFromNode()
            nbr_id = from_node.getID()
            if nbr_id in seen or nbr_id == node.getID():
                continue
            if nbr_id not in self._id_to_idx:
                continue
            seen.add(nbr_id)

            dist = edge.getLength()
            n_lanes = edge.getLaneNumber()
            neighbors.append((nbr_id, dist, n_lanes))

        return neighbors

    def _get_neighbor_tl_ids_for_tls(self, tl_id: str) -> list[tuple[str, float, int]]:
        """Return ``(neighbor_tl_id, distance, n_lanes)`` for a TLS controller.

        In some SUMO maps (for example Berlin), traffic-light IDs returned by
        TraCI are controller IDs that do not match a node ID. For those maps we
        derive neighbours from nodes controlled by each TLS.
        """
        controlled_nodes = self._tls_nodes_by_id.get(tl_id)
        if controlled_nodes:
            neighbors: list[tuple[str, float, int]] = []
            seen: set[str] = set()
            for node in controlled_nodes:
                for edge in node.getOutgoing():
                    to_node = edge.getToNode()
                    nbr_id = self._node_to_tls_id.get(to_node.getID())
                    if nbr_id is None or nbr_id == tl_id or nbr_id in seen:
                        continue
                    if nbr_id not in self._id_to_idx:
                        continue
                    seen.add(nbr_id)
                    neighbors.append((nbr_id, edge.getLength(), edge.getLaneNumber()))

                for edge in node.getIncoming():
                    from_node = edge.getFromNode()
                    nbr_id = self._node_to_tls_id.get(from_node.getID())
                    if nbr_id is None or nbr_id == tl_id or nbr_id in seen:
                        continue
                    if nbr_id not in self._id_to_idx:
                        continue
                    seen.add(nbr_id)
                    neighbors.append((nbr_id, edge.getLength(), edge.getLaneNumber()))
            return neighbors

        # Fallback for scenarios where TLS IDs equal node IDs.
        try:
            node = self.net.getNode(tl_id)
        except KeyError:
            return []
        return self._get_neighbor_tl_ids(node)

    def _build_tls_node_map(self) -> dict[str, list[Any]]:
        """Map TLS/controller ID -> list of controlled SUMO nodes."""
        if not hasattr(self.net, "getTrafficLights"):
            return {}

        tls_nodes_by_id: dict[str, list[Any]] = {}
        for tls in self.net.getTrafficLights():
            tls_id = tls.getID()
            nodes_by_id: dict[str, Any] = {}

            # sumolib stores TLS links as [fromLane, toLane, tlLinkIndex]
            for conn in tls.getConnections():
                if len(conn) < 2:
                    continue
                from_lane = conn[0]
                to_lane = conn[1]
                for node in (
                    from_lane.getEdge().getToNode(),
                    to_lane.getEdge().getFromNode(),
                ):
                    nodes_by_id[node.getID()] = node

            if nodes_by_id:
                tls_nodes_by_id[tls_id] = list(nodes_by_id.values())

        return tls_nodes_by_id

    @staticmethod
    def _build_node_to_tls_map(tls_nodes_by_id: dict[str, list[Any]]) -> dict[str, str]:
        """Map SUMO node ID -> TLS/controller ID (first assignment wins)."""
        node_to_tls_id: dict[str, str] = {}
        for tls_id, nodes in tls_nodes_by_id.items():
            for node in nodes:
                node_id = node.getID()
                if node_id not in node_to_tls_id:
                    node_to_tls_id[node_id] = tls_id
        return node_to_tls_id
