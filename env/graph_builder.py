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

        tl_nodes = {
            tl_id: self.net.getNode(tl_id) for tl_id in self.tl_ids
        }

        for tl_id, node in tl_nodes.items():
            src_idx = self._id_to_idx[tl_id]
            neighbors = self._get_neighbor_tl_ids(node)

            for nbr_id, dist, n_lanes in neighbors:
                if nbr_id not in self._id_to_idx:
                    continue
                dst_idx = self._id_to_idx[nbr_id]
                # Undirected: add both directions
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
