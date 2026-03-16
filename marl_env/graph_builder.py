"""Build the intersection adjacency graph from a SUMO network.

The graph is built once at environment init and stays fixed during training.
Node features change every step; topology does not.
"""

from __future__ import annotations

import heapq
from typing import Any, Literal

import torch
from torch import Tensor

try:
    import sumolib  # type: ignore[import-untyped]
except ImportError:
    sumolib = None  # allow import without sumolib for typing


GraphConstructionMode = Literal[
    "original",
    "walk_to_light",
    "all_intersections",
]
GRAPH_BUILDER_MODES: tuple[GraphConstructionMode, ...] = (
    "original",
    "walk_to_light",
    "all_intersections",
)


class GraphBuilder:
    """Builds a homogeneous intersection graph from a SUMO ``.net.xml``.

    Nodes  = controlled traffic-light intersections.
    Edges  = pairs of intersections connected by road segments, allowing
    traversal through intermediate non-signalized SUMO nodes until the next
    controlled intersection is reached.

    Optional edge attributes include distance between intersections and
    the number of connecting lanes.
    """

    def __init__(
        self,
        net_file: str,
        tl_ids: list[str],
        *,
        mode: GraphConstructionMode = "original",
    ) -> None:
        if sumolib is None:
            raise ImportError("sumolib is required — install SUMO tools.")
        if mode not in GRAPH_BUILDER_MODES:
            raise ValueError(
                f"Unknown graph builder mode {mode!r}. Expected one of {GRAPH_BUILDER_MODES}."
            )

        self.net: Any = sumolib.net.readNet(net_file, withInternal=False)
        self.mode = mode
        self.tl_ids = tl_ids
        self._id_to_idx = {tl_id: i for i, tl_id in enumerate(tl_ids)}
        self._tls_nodes_by_id = self._build_tls_node_map()
        self._node_to_tls_id = self._build_node_to_tls_map(self._tls_nodes_by_id)
        self._graph_node_ids = self._build_graph_node_ids()
        self._graph_id_to_idx = {
            node_id: i for i, node_id in enumerate(self._graph_node_ids)
        }
        self._agent_node_id_lists = self._build_agent_node_id_lists()
        self._attached_rl_ids_by_node = self._build_attached_rl_ids_by_node()

        self._edge_index: Tensor | None = None
        self._edge_attr: Tensor | None = None
        self._node_positions: Tensor | None = None
        self._agent_node_indices: Tensor | None = None
        self._agent_node_mask: Tensor | None = None

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

        if self.mode == "all_intersections":
            for node_id in self.node_ids:
                src_idx = self._graph_id_to_idx[node_id]
                try:
                    node = self.net.getNode(node_id)
                except KeyError:
                    continue
                neighbors = self._get_immediate_neighbor_node_ids(node)

                for nbr_id, dist, n_lanes in neighbors:
                    dst_idx = self._graph_id_to_idx[nbr_id]
                    src_list.append(src_idx)
                    dst_list.append(dst_idx)
                    attrs.append([dist, float(n_lanes)])
        else:
            for tl_id in self.tl_ids:
                src_idx = self._id_to_idx[tl_id]
                if self.mode == "original":
                    neighbors = self._get_immediate_neighbor_tl_ids_for_tls(tl_id)
                else:
                    neighbors = self._get_neighbor_tl_ids_for_tls(tl_id)

                for nbr_id, dist, n_lanes in neighbors:
                    if nbr_id not in self._id_to_idx:
                        continue
                    dst_idx = self._id_to_idx[nbr_id]
                    # One direction per iteration; the reverse is added
                    # when the neighbour is processed as the source.
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
        return len(self._graph_node_ids)

    @property
    def node_ids(self) -> list[str]:
        return list(self._graph_node_ids)

    @property
    def node_positions(self) -> Tensor:
        """Return ordered TLS/controller positions as ``[num_nodes, 2]``.

        Positions align with ``self.tl_ids``. For controller-style TLS IDs,
        the position is the centroid of the controlled SUMO node coordinates.
        """
        if self._node_positions is None:
            coords = [self._get_position_for_node(node_id) for node_id in self._graph_node_ids]
            self._node_positions = torch.tensor(coords, dtype=torch.float32)
        return self._node_positions

    @property
    def agent_node_indices(self) -> Tensor:
        if self._agent_node_indices is None or self._agent_node_mask is None:
            self._build_agent_node_index_tensors()
        assert self._agent_node_indices is not None
        return self._agent_node_indices

    @property
    def agent_node_mask(self) -> Tensor:
        if self._agent_node_indices is None or self._agent_node_mask is None:
            self._build_agent_node_index_tensors()
        assert self._agent_node_mask is not None
        return self._agent_node_mask

    @property
    def attached_rl_ids_by_node(self) -> list[tuple[str, ...]]:
        return list(self._attached_rl_ids_by_node)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_immediate_neighbor_tl_ids(
        self, node: Any
    ) -> list[tuple[str, float, int]]:
        """Return immediate ``(neighbor_tl_id, distance, n_lanes)`` tuples."""
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
            neighbors.append((nbr_id, float(edge.getLength()), int(edge.getLaneNumber())))

        for edge in node.getIncoming():
            from_node = edge.getFromNode()
            nbr_id = from_node.getID()
            if nbr_id in seen or nbr_id == node.getID():
                continue
            if nbr_id not in self._id_to_idx:
                continue
            seen.add(nbr_id)
            neighbors.append((nbr_id, float(edge.getLength()), int(edge.getLaneNumber())))

        return neighbors

    def _get_neighbor_tl_ids(
        self, node: Any
    ) -> list[tuple[str, float, int]]:
        """Return walk-contracted ``(neighbor_tl_id, distance, n_lanes)`` tuples."""
        return self._find_neighbor_tls(source_tl_id=node.getID(), start_nodes=[node])

    def _get_immediate_neighbor_tl_ids_for_tls(
        self, tl_id: str
    ) -> list[tuple[str, float, int]]:
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
                    neighbors.append(
                        (nbr_id, float(edge.getLength()), int(edge.getLaneNumber()))
                    )

                for edge in node.getIncoming():
                    from_node = edge.getFromNode()
                    nbr_id = self._node_to_tls_id.get(from_node.getID())
                    if nbr_id is None or nbr_id == tl_id or nbr_id in seen:
                        continue
                    if nbr_id not in self._id_to_idx:
                        continue
                    seen.add(nbr_id)
                    neighbors.append(
                        (nbr_id, float(edge.getLength()), int(edge.getLaneNumber()))
                    )
            return neighbors

        try:
            node = self.net.getNode(tl_id)
        except KeyError:
            return []
        return self._get_immediate_neighbor_tl_ids(node)

    def _get_neighbor_tl_ids_for_tls(self, tl_id: str) -> list[tuple[str, float, int]]:
        """Return ``(neighbor_tl_id, distance, n_lanes)`` for a TLS controller.

        In some SUMO maps (for example Berlin), traffic-light IDs returned by
        TraCI are controller IDs that do not match a node ID. For those maps we
        derive neighbours from nodes controlled by each TLS.
        """
        controlled_nodes = self._tls_nodes_by_id.get(tl_id)
        if controlled_nodes:
            return self._find_neighbor_tls(
                source_tl_id=tl_id,
                start_nodes=controlled_nodes,
            )

        # Fallback for scenarios where TLS IDs equal node IDs.
        try:
            node = self.net.getNode(tl_id)
        except KeyError:
            return []
        return self._get_neighbor_tl_ids(node)

    def _get_immediate_neighbor_node_ids(
        self,
        node: Any,
    ) -> list[tuple[str, float, int]]:
        neighbors: list[tuple[str, float, int]] = []
        seen: set[str] = set()

        for edge in node.getOutgoing():
            to_node = edge.getToNode()
            nbr_id = to_node.getID()
            if nbr_id in seen or nbr_id == node.getID():
                continue
            if nbr_id not in self._graph_id_to_idx:
                continue
            seen.add(nbr_id)
            neighbors.append((nbr_id, float(edge.getLength()), int(edge.getLaneNumber())))

        for edge in node.getIncoming():
            from_node = edge.getFromNode()
            nbr_id = from_node.getID()
            if nbr_id in seen or nbr_id == node.getID():
                continue
            if nbr_id not in self._graph_id_to_idx:
                continue
            seen.add(nbr_id)
            neighbors.append((nbr_id, float(edge.getLength()), int(edge.getLaneNumber())))

        return neighbors

    @staticmethod
    def _iter_neighbor_nodes(node: Any) -> list[tuple[Any, float, int]]:
        """Return adjacent SUMO nodes with edge distance and lane count."""
        neighbors: list[tuple[Any, float, int]] = []

        for edge in node.getOutgoing():
            to_node = edge.getToNode()
            if to_node.getID() == node.getID():
                continue
            neighbors.append((to_node, float(edge.getLength()), int(edge.getLaneNumber())))

        for edge in node.getIncoming():
            from_node = edge.getFromNode()
            if from_node.getID() == node.getID():
                continue
            neighbors.append((from_node, float(edge.getLength()), int(edge.getLaneNumber())))

        return neighbors

    def _find_neighbor_tls(
        self,
        source_tl_id: str,
        start_nodes: list[Any],
    ) -> list[tuple[str, float, int]]:
        """Walk through non-TLS nodes until the next controlled TLS is found.

        The returned lane count is the minimum lane count along the selected
        shortest path, which acts as a simple bottleneck estimate.
        """
        if not start_nodes:
            return []

        heap: list[tuple[float, float, str, Any]] = []
        best_node_distance: dict[str, float] = {}
        best_terminal: dict[str, tuple[float, int]] = {}

        for node in start_nodes:
            node_id = node.getID()
            best_node_distance[node_id] = 0.0
            heapq.heappush(heap, (0.0, float("inf"), node_id, node))

        while heap:
            dist, bottleneck_lanes, node_id, node = heapq.heappop(heap)
            if dist > best_node_distance.get(node_id, float("inf")):
                continue

            terminal_tl_id = self._node_to_tls_id.get(node_id)
            if terminal_tl_id is None and node_id in self._id_to_idx:
                terminal_tl_id = node_id
            if terminal_tl_id is not None and terminal_tl_id != source_tl_id:
                lane_count = 0 if bottleneck_lanes == float("inf") else int(bottleneck_lanes)
                prev = best_terminal.get(terminal_tl_id)
                if prev is None or dist < prev[0] or (
                    dist == prev[0] and lane_count > prev[1]
                ):
                    best_terminal[terminal_tl_id] = (dist, lane_count)
                continue

            for next_node, edge_dist, edge_lanes in self._iter_neighbor_nodes(node):
                next_id = next_node.getID()
                next_dist = dist + edge_dist
                prev_best = best_node_distance.get(next_id)
                if prev_best is not None and next_dist >= prev_best:
                    continue

                best_node_distance[next_id] = next_dist
                next_bottleneck = (
                    float(edge_lanes)
                    if bottleneck_lanes == float("inf")
                    else min(bottleneck_lanes, float(edge_lanes))
                )
                heapq.heappush(heap, (next_dist, next_bottleneck, next_id, next_node))

        return [
            (tl_id, distance, lane_count)
            for tl_id, (distance, lane_count) in best_terminal.items()
            if tl_id in self._id_to_idx
        ]

    def _build_graph_node_ids(self) -> list[str]:
        if self.mode == "all_intersections":
            return [node.getID() for node in self.net.getNodes()]
        return list(self.tl_ids)

    def _build_agent_node_id_lists(self) -> list[list[str]]:
        agent_node_ids: list[list[str]] = []

        for tl_id in self.tl_ids:
            if self.mode == "all_intersections":
                controlled_nodes = self._tls_nodes_by_id.get(tl_id)
                if controlled_nodes:
                    raw_ids = [node.getID() for node in controlled_nodes]
                elif tl_id in self._graph_id_to_idx:
                    raw_ids = [tl_id]
                else:
                    raw_ids = []
            else:
                raw_ids = [tl_id]

            deduped = [
                node_id
                for node_id in dict.fromkeys(raw_ids)
                if node_id in self._graph_id_to_idx
            ]
            if not deduped:
                raise ValueError(
                    f"Could not map traffic-light/controller {tl_id!r} to any graph node "
                    f"under graph mode {self.mode!r}."
                )
            agent_node_ids.append(deduped)

        return agent_node_ids

    def _build_attached_rl_ids_by_node(self) -> list[tuple[str, ...]]:
        node_to_rl_ids: dict[str, list[str]] = {
            node_id: [] for node_id in self._graph_node_ids
        }
        for tl_id, node_ids in zip(self.tl_ids, self._agent_node_id_lists, strict=True):
            for node_id in node_ids:
                node_to_rl_ids.setdefault(node_id, []).append(tl_id)
        return [
            tuple(node_to_rl_ids.get(node_id, []))
            for node_id in self._graph_node_ids
        ]

    def _build_agent_node_index_tensors(self) -> None:
        max_nodes_per_agent = max(len(node_ids) for node_ids in self._agent_node_id_lists)
        indices = torch.full(
            (len(self.tl_ids), max_nodes_per_agent),
            -1,
            dtype=torch.long,
        )
        mask = torch.zeros((len(self.tl_ids), max_nodes_per_agent), dtype=torch.bool)

        for agent_idx, node_ids in enumerate(self._agent_node_id_lists):
            for slot, node_id in enumerate(node_ids):
                indices[agent_idx, slot] = self._graph_id_to_idx[node_id]
                mask[agent_idx, slot] = True

        self._agent_node_indices = indices
        self._agent_node_mask = mask

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

    @staticmethod
    def _centroid(nodes: list[Any]) -> tuple[float, float]:
        if not nodes:
            return float("nan"), float("nan")

        xs: list[float] = []
        ys: list[float] = []
        for node in nodes:
            x, y = node.getCoord()
            xs.append(float(x))
            ys.append(float(y))

        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _get_position_for_tls(self, tl_id: str) -> tuple[float, float]:
        controlled_nodes = self._tls_nodes_by_id.get(tl_id)
        if controlled_nodes:
            return self._centroid(controlled_nodes)

        try:
            node = self.net.getNode(tl_id)
        except KeyError:
            return float("nan"), float("nan")

        x, y = node.getCoord()
        return float(x), float(y)

    def _get_position_for_node(self, node_id: str) -> tuple[float, float]:
        if self.mode == "all_intersections":
            try:
                node = self.net.getNode(node_id)
            except KeyError:
                return float("nan"), float("nan")
            x, y = node.getCoord()
            return float(x), float(y)
        return self._get_position_for_tls(node_id)
