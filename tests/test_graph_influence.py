from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import get_num_hops

from config_utils import resolve_repo_path
from models.graph_encoder import GraphEncoder
from models.local_neighbor_gat_discrete_sac import LocalNeighborGATDiscreteSAC
from visualization.graph_influence import (
    EncoderInfluenceModel,
    _total_influence_with_edge_attr,
    deduplicate_undirected_edges,
    extract_road_segments,
)


def test_deduplicate_undirected_edges_collapses_reciprocal_pairs() -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
        dtype=torch.long,
    )

    assert deduplicate_undirected_edges(edge_index) == [(0, 1), (1, 2), (0, 2)]


def test_encoder_influence_model_uses_fixed_edge_attr() -> None:
    torch.manual_seed(7)
    encoder = GraphEncoder(
        in_dim=3,
        hidden_dim=8,
        out_dim=4,
        heads=2,
        edge_dim=2,
        dropout=0.0,
    )
    encoder.eval()

    x = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]],
        dtype=torch.float32,
    )

    wrapper = EncoderInfluenceModel(encoder, edge_attr=edge_attr)

    expected = encoder(x, edge_index, edge_attr)
    actual = wrapper(x, edge_index)

    assert torch.allclose(actual, expected)
    assert get_num_hops(wrapper) == 2


def test_total_influence_handles_empty_edge_graph() -> None:
    class DummyNodeModel(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
        ) -> torch.Tensor:
            _ = edge_index
            _ = edge_attr
            return x

    data = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=3,
    )

    curve, breadth = _total_influence_with_edge_attr(
        DummyNodeModel(),
        data,
        max_hops=2,
        num_samples=3,
        device="cpu",
        seed=0,
    )

    assert curve.shape == (3,)
    assert torch.allclose(curve, torch.tensor([1.0, 0.0, 0.0]))
    assert breadth == pytest.approx(0.0)


def test_extract_road_segments_skips_internal_edges() -> None:
    class FakeEdge:
        def __init__(self, edge_id: str, shape: list[tuple[float, float]], lanes: int) -> None:
            self._id = edge_id
            self._shape = shape
            self._lanes = lanes

        def getID(self) -> str:
            return self._id

        def getShape(self) -> list[tuple[float, float]]:
            return self._shape

        def getLaneNumber(self) -> int:
            return self._lanes

    class FakeNet:
        def getEdges(self) -> list[FakeEdge]:
            return [
                FakeEdge("a", [(0.0, 0.0), (1.0, 1.0)], 2),
                FakeEdge(":internal", [(1.0, 1.0), (2.0, 2.0)], 1),
                FakeEdge("b", [(2.0, 2.0)], 1),
                FakeEdge("c", [(3.0, 3.0), (4.0, 4.0), (5.0, 4.0)], 1),
            ]

    segments = extract_road_segments(FakeNet())

    assert len(segments) == 2
    assert segments[0].lanes == 2
    assert segments[0].points == ((0.0, 0.0), (1.0, 1.0))
    assert segments[1].points[-1] == (5.0, 4.0)


def test_agent_pooling_averages_multiple_graph_nodes_per_rl_agent() -> None:
    z_nodes = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 2.0],
            [10.0, 4.0],
        ],
        dtype=torch.float32,
    )
    agent_node_indices = torch.tensor([[0, 1], [2, -1]], dtype=torch.long)
    agent_node_mask = torch.tensor([[True, True], [True, False]])

    pooled = LocalNeighborGATDiscreteSAC._pool_agent_latents(
        z_nodes,
        agent_node_indices=agent_node_indices,
        agent_node_mask=agent_node_mask,
    )

    assert torch.allclose(
        pooled,
        torch.tensor([[2.0, 1.0], [10.0, 4.0]], dtype=torch.float32),
    )


@pytest.mark.sumo
def test_run_visualization_grid4x4_smoke(
    tmp_path: Path,
    sumo_stack: dict[str, object],
) -> None:
    _ = sumo_stack
    checkpoint_path = resolve_repo_path("runs/grid4x4_meta_critic/best_agent.pt")
    if not checkpoint_path.exists():
        pytest.skip("Missing grid4x4 checkpoint for visualization smoke test.")

    out_dir = tmp_path / "visualization"

    cmd = [
        sys.executable,
        str(resolve_repo_path("scripts/visualize_graph_influence.py")),
        "env=grid4x4",
        f"runtime.checkpoint_path={checkpoint_path}",
        f"runtime.out_dir={out_dir}",
        "analysis.num_snapshots=2",
        "analysis.curve_num_samples=4",
        "analysis.map_num_samples=4",
    ]
    subprocess.run(cmd, check=True, cwd=resolve_repo_path("."))

    assert (out_dir / "graph_topology.png").exists()
    assert (out_dir / "influence_curve.png").exists()
    assert (out_dir / "influence_map.png").exists()
    assert (out_dir / "influence_summary.json").exists()
    assert (out_dir / "node_influence.csv").exists()
