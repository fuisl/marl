"""Graph topology and City-Networks-style influence visualization helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from torch.autograd.functional import jacobian
from torch_geometric.data import Data
from torch_geometric.utils import get_num_hops, k_hop_subgraph
from torch_geometric.utils.influence import k_hop_subsets_exact

from marl_env.sumo_env import TrafficSignalEnv
from models.marl_discrete_sac import MARLDiscreteSAC


_PYG_RUNTIME_HINT = (
    "PyG influence analysis failed because optional compiled extensions appear "
    "to be mismatched with the installed Torch/CUDA build. Reinstall matching "
    "PyG wheels in the active environment before rerunning visualization."
)


@dataclass(frozen=True)
class SnapshotGraph:
    """A single traffic-state graph snapshot used for influence analysis."""

    step_index: int
    data: Data


@dataclass(frozen=True)
class RoadSegment:
    """A road polyline extracted from the SUMO network."""

    points: tuple[tuple[float, float], ...]
    lanes: int


class EncoderInfluenceModel(nn.Module):
    """Wrap the trained graph encoder with fixed edge attributes."""

    def __init__(self, encoder: nn.Module, edge_attr: Tensor | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        if edge_attr is None:
            self.edge_attr = None
        else:
            self.register_buffer("edge_attr", edge_attr.detach().clone())

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        edge_attr_to_use = self.edge_attr if edge_attr is None else edge_attr
        return self.encoder(x, edge_index, edge_attr_to_use)


def deduplicate_undirected_edges(edge_index: Tensor) -> list[tuple[int, int]]:
    """Collapse reciprocal directed edges into a stable undirected view."""
    unique_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for src, dst in edge_index.t().tolist():
        edge = (min(int(src), int(dst)), max(int(src), int(dst)))
        if edge in seen:
            continue
        seen.add(edge)
        unique_edges.append(edge)

    return unique_edges


def extract_road_segments(net: Any) -> list[RoadSegment]:
    """Extract non-internal road polylines from a SUMO network."""
    segments: list[RoadSegment] = []

    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(":"):
            continue

        shape = edge.getShape()
        if len(shape) < 2:
            continue

        segments.append(
            RoadSegment(
                points=tuple((float(x), float(y)) for x, y in shape),
                lanes=max(1, int(edge.getLaneNumber())),
            )
        )

    return segments


def resolve_max_hops(model: nn.Module, configured_max_hops: int | None) -> int:
    if configured_max_hops is not None:
        return int(configured_max_hops)
    return int(get_num_hops(model))


def resolve_curve_num_samples(num_nodes: int, configured_num_samples: int | None) -> int:
    if configured_num_samples is not None:
        return min(int(configured_num_samples), num_nodes)
    if num_nodes <= 256:
        return num_nodes
    return min(128, num_nodes)


def resolve_map_num_samples(num_nodes: int, configured_num_samples: int | None) -> int:
    if configured_num_samples is not None:
        return min(int(configured_num_samples), num_nodes)
    if num_nodes <= 256:
        return num_nodes
    return min(512, num_nodes)


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Return evenly spaced indices across ``range(length)``."""
    if length <= 0:
        raise ValueError("Episode length must be positive.")

    count = max(1, min(int(count), length))
    if count == 1:
        return [length // 2]

    return [
        round(i * (length - 1) / (count - 1))
        for i in range(count)
    ]


def select_sampled_nodes(
    num_nodes: int,
    num_samples: int,
    *,
    seed: int = 0,
) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.randperm(num_nodes, generator=generator)[:num_samples]
    return sorted(int(i) for i in sampled.tolist())


def receptive_field_breadth(influence_per_hop: Tensor) -> float:
    total = float(influence_per_hop.sum().item())
    if total <= 0.0:
        return 0.0

    hops = torch.arange(
        influence_per_hop.numel(),
        dtype=influence_per_hop.dtype,
        device=influence_per_hop.device,
    )
    weights = influence_per_hop / influence_per_hop.sum()
    return float((weights * hops).sum().item())


def _raise_with_pyg_hint(exc: Exception) -> None:
    message = str(exc)
    if isinstance(exc, (ImportError, OSError)) or any(
        token in message
        for token in ("undefined symbol", "_version_cuda", "Could not load this library")
    ):
        raise RuntimeError(_PYG_RUNTIME_HINT) from exc
    raise exc


def _extract_edge_attr(td: TensorDict) -> Tensor | None:
    if "edge_attr" not in td.keys():
        return None
    return td["edge_attr"].detach().cpu()


def _jacobian_l1_safe(
    model: nn.Module,
    data: Data,
    max_hops: int,
    node_idx: int,
    device: torch.device | str,
    *,
    vectorize: bool = True,
) -> Tensor:
    """Jacobian helper that preserves edge attributes and explicit node count."""
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    x = data.x
    num_nodes = int(data.num_nodes)

    k_hop_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx,
        max_hops,
        edge_index,
        num_nodes=num_nodes,
        relabel_nodes=True,
    )
    root_pos = int(mapping[0])

    device = torch.device(device)
    sub_x = x[k_hop_nodes].to(device)
    sub_edge_index = sub_edge_index.to(device)
    sub_edge_attr = edge_attr[edge_mask].to(device) if edge_attr is not None else None
    model = model.to(device)

    def _forward(x_in: Tensor) -> Tensor:
        return model(x_in, sub_edge_index, sub_edge_attr)[root_pos]

    jac = jacobian(_forward, sub_x, vectorize=vectorize)
    influence_sub = jac.abs().sum(dim=(0, 2))

    influence_full = torch.zeros(
        num_nodes,
        dtype=influence_sub.dtype,
        device=device,
    )
    influence_full[k_hop_nodes] = influence_sub
    return influence_full


def _jacobian_l1_agg_per_hop_safe(
    model: nn.Module,
    data: Data,
    max_hops: int,
    node_idx: int,
    device: torch.device | str,
    *,
    vectorize: bool = True,
) -> Tensor:
    influence = _jacobian_l1_safe(
        model,
        data,
        max_hops,
        node_idx,
        device,
        vectorize=vectorize,
    )
    hop_subsets = k_hop_subsets_exact(
        node_idx,
        max_hops,
        data.edge_index,
        int(data.num_nodes),
        influence.device,
    )
    return torch.stack([influence[subset].sum() for subset in hop_subsets])


def _total_influence_with_edge_attr(
    model: nn.Module,
    data: Data,
    *,
    max_hops: int,
    num_samples: int,
    device: torch.device | str,
    seed: int = 0,
    normalize: bool = True,
    average: bool = True,
    vectorize: bool = True,
) -> tuple[Tensor, float]:
    sampled_nodes = select_sampled_nodes(int(data.num_nodes), num_samples, seed=seed)
    influence_rows = [
        _jacobian_l1_agg_per_hop_safe(
            model,
            data,
            max_hops,
            node_idx,
            device,
            vectorize=vectorize,
        ).detach().cpu()
        for node_idx in sampled_nodes
    ]
    all_nodes = torch.vstack(influence_rows)

    if average:
        avg_influence = all_nodes.mean(dim=0)
        if normalize and float(avg_influence[0].item()) != 0.0:
            avg_influence = avg_influence / avg_influence[0]
    else:
        avg_influence = all_nodes

    breadths = [receptive_field_breadth(row) for row in all_nodes]
    avg_breadth = float(sum(breadths) / len(breadths))
    return avg_influence, avg_breadth


@torch.no_grad()
def count_episode_steps(
    env: TrafficSignalEnv,
    agent: MARLDiscreteSAC,
    device: torch.device | str,
) -> int:
    """Count deterministic decision steps in one episode."""
    td = env.reset().to(device)
    steps = 0

    while True:
        actions, _ = agent.select_action(
            td["agents", "observation"],
            td["edge_index"],
            td.get("edge_attr", None),
            td["agents", "action_mask"],
            deterministic=True,
        )

        next_td = env.step(actions.cpu()).to(device)
        steps += 1
        if next_td["done"].item():
            break
        td = next_td

    return steps


@torch.no_grad()
def collect_snapshot_graphs(
    env: TrafficSignalEnv,
    agent: MARLDiscreteSAC,
    device: torch.device | str,
    snapshot_steps: list[int],
) -> list[SnapshotGraph]:
    """Collect graph snapshots at selected decision steps."""
    wanted = set(snapshot_steps)
    snapshots: list[SnapshotGraph] = []
    td = env.reset().to(device)
    step_index = 0

    while True:
        if step_index in wanted:
            snapshots.append(
                SnapshotGraph(
                    step_index=step_index,
                    data=Data(
                        x=td["agents", "observation"].detach().cpu(),
                        edge_index=td["edge_index"].detach().cpu(),
                        edge_attr=_extract_edge_attr(td),
                        num_nodes=int(td["agents", "observation"].shape[0]),
                    ),
                )
            )

        actions, _ = agent.select_action(
            td["agents", "observation"],
            td["edge_index"],
            td.get("edge_attr", None),
            td["agents", "action_mask"],
            deterministic=True,
        )

        next_td = env.step(actions.cpu()).to(device)
        if next_td["done"].item():
            break
        td = next_td
        step_index += 1

    if len(snapshots) != len(snapshot_steps):
        raise RuntimeError(
            f"Expected {len(snapshot_steps)} snapshots, collected {len(snapshots)}."
        )

    return snapshots


def load_agent_for_visualization(
    checkpoint_path: Path,
    model_cfg: dict[str, Any],
    td0: TensorDict,
    num_actions: int,
    device: torch.device,
) -> MARLDiscreteSAC:
    obs_dim = int(td0["agents", "observation"].shape[-1])

    agent = MARLDiscreteSAC(
        obs_dim=obs_dim,
        num_actions=num_actions,
        encoder_cfg=dict(model_cfg.get("encoder_cfg", {})),
        actor_cfg=dict(model_cfg.get("actor_cfg", {})),
        critic_cfg=dict(model_cfg.get("critic_cfg", {})),
        init_alpha=float(model_cfg.get("init_alpha", 0.2)),
        tau=float(model_cfg.get("tau", 0.005)),
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def _plot_edges(
    ax: plt.Axes,
    positions: Tensor,
    edges: list[tuple[int, int]],
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    valid = torch.isfinite(positions).all(dim=1)
    for src, dst in edges:
        if not (valid[src] and valid[dst]):
            continue
        xs = [float(positions[src, 0]), float(positions[dst, 0])]
        ys = [float(positions[src, 1]), float(positions[dst, 1])]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=1)


def _plot_road_segments(ax: plt.Axes, road_segments: list[RoadSegment]) -> None:
    _plot_road_segments_with_style(
        ax,
        road_segments,
        color="#111111",
        alpha=0.92,
        base_linewidth=0.6,
        linewidth_scale=2.0,
        zorder=0,
    )


def _plot_road_segments_with_style(
    ax: plt.Axes,
    road_segments: list[RoadSegment],
    *,
    color: str,
    alpha: float,
    base_linewidth: float,
    linewidth_scale: float,
    zorder: int,
) -> None:
    if not road_segments:
        return

    max_lanes = max(segment.lanes for segment in road_segments)
    for segment in road_segments:
        xs = [pt[0] for pt in segment.points]
        ys = [pt[1] for pt in segment.points]
        lane_scale = segment.lanes / max_lanes
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=base_linewidth + linewidth_scale * lane_scale,
            alpha=alpha,
            solid_capstyle="round",
            zorder=zorder,
        )


def _compute_plot_bounds(
    positions: Tensor,
    road_segments: list[RoadSegment] | None = None,
) -> tuple[float, float, float, float] | None:
    x_values: list[float] = []
    y_values: list[float] = []

    valid = torch.isfinite(positions).all(dim=1)
    if bool(valid.any()):
        coords = positions[valid]
        x_values.extend(float(v) for v in coords[:, 0].tolist())
        y_values.extend(float(v) for v in coords[:, 1].tolist())

    for segment in road_segments or []:
        x_values.extend(pt[0] for pt in segment.points)
        y_values.extend(pt[1] for pt in segment.points)

    if not x_values or not y_values:
        return None

    return min(x_values), max(x_values), min(y_values), max(y_values)


def _style_map_axes(
    ax: plt.Axes,
    positions: Tensor,
    road_segments: list[RoadSegment] | None = None,
) -> None:
    bounds = _compute_plot_bounds(positions, road_segments)
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        x_pad = max((x_max - x_min) * 0.05, 1.0)
        y_pad = max((y_max - y_min) * 0.05, 1.0)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_graph_topology(
    out_path: Path,
    positions: Tensor,
    edge_index: Tensor,
    tl_ids: list[str],
    road_segments: list[RoadSegment] | None = None,
) -> None:
    edges = deduplicate_undirected_edges(edge_index)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#F5F6F7")

    _plot_road_segments_with_style(
        ax,
        road_segments or [],
        color="#C6CCD3",
        alpha=0.95,
        base_linewidth=0.45,
        linewidth_scale=1.2,
        zorder=0,
    )
    # Add a pale halo first so the highlighted graph edges remain visible
    # even when they overlap dense road geometry.
    _plot_edges(ax, positions, edges, color="#FDFDFD", linewidth=3.8, alpha=0.95)
    _plot_edges(ax, positions, edges, color="#0069C0", linewidth=2.2, alpha=0.98)

    valid = torch.isfinite(positions).all(dim=1)
    node_size = 70 if len(tl_ids) <= 100 else 12
    ax.scatter(
        positions[valid, 0].tolist(),
        positions[valid, 1].tolist(),
        s=node_size,
        c="#0057A8",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    if len(tl_ids) <= 100:
        for idx, tl_id in enumerate(tl_ids):
            if not bool(valid[idx]):
                continue
            ax.text(
                float(positions[idx, 0]),
                float(positions[idx, 1]),
                tl_id,
                fontsize=6,
                ha="center",
                va="bottom",
                zorder=3,
            )

    ax.set_title(
        f"Intersection Graph Topology ({len(tl_ids)} nodes, {len(edges)} undirected edges)"
    )
    _style_map_axes(ax, positions, road_segments)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_influence_curve(
    out_path: Path,
    avg_curve: Tensor,
    snapshot_curves: list[Tensor],
    avg_receptive_field: float,
) -> None:
    hops = list(range(avg_curve.numel()))
    stacked = torch.stack(snapshot_curves)
    std_curve = stacked.std(dim=0) if len(snapshot_curves) > 1 else torch.zeros_like(avg_curve)

    fig, ax = plt.subplots(figsize=(8, 5))
    for curve in snapshot_curves:
        ax.plot(hops, curve.tolist(), color="#C9CED4", linewidth=1.0, alpha=0.9)

    ax.fill_between(
        hops,
        (avg_curve - std_curve).tolist(),
        (avg_curve + std_curve).tolist(),
        color="#6BAED6",
        alpha=0.2,
    )
    ax.plot(hops, avg_curve.tolist(), color="#0B5394", linewidth=2.5, marker="o")
    ax.set_xlabel("Hop distance")
    ax.set_ylabel("Normalized total influence")
    ax.set_title(f"Total Influence by Hop (Average R = {avg_receptive_field:.3f})")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_node_influence_map(
    out_path: Path,
    positions: Tensor,
    edge_index: Tensor,
    tl_ids: list[str],
    node_rows: list[dict[str, Any]],
    road_segments: list[RoadSegment] | None = None,
) -> None:
    edges = deduplicate_undirected_edges(edge_index)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#F5F6F7")

    _plot_road_segments_with_style(
        ax,
        road_segments or [],
        color="#C6CCD3",
        alpha=0.95,
        base_linewidth=0.45,
        linewidth_scale=1.2,
        zorder=0,
    )
    _plot_edges(ax, positions, edges, color="#FDFDFD", linewidth=3.4, alpha=0.92)
    _plot_edges(ax, positions, edges, color="#0069C0", linewidth=1.7, alpha=0.72)

    sampled_rows = [row for row in node_rows if row["is_sampled"]]
    unsampled_rows = [row for row in node_rows if not row["is_sampled"]]

    if unsampled_rows:
        xs = [row["x"] for row in unsampled_rows if row["has_position"]]
        ys = [row["y"] for row in unsampled_rows if row["has_position"]]
        if xs:
            ax.scatter(
                xs,
                ys,
                s=12,
                c="#C7CED6",
                edgecolors="white",
                linewidths=0.25,
                alpha=0.95,
                zorder=2,
            )

    if sampled_rows:
        xs = [row["x"] for row in sampled_rows if row["has_position"]]
        ys = [row["y"] for row in sampled_rows if row["has_position"]]
        vals = [row["receptive_field_breadth"] for row in sampled_rows if row["has_position"]]
        scatter = ax.scatter(
            xs,
            ys,
            s=16 if len(tl_ids) > 100 else 55,
            c=vals,
            cmap="viridis",
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Influence-weighted receptive field $R_i$")

    if len(tl_ids) <= 100:
        for row in sampled_rows:
            if not row["has_position"]:
                continue
            ax.text(
                row["x"],
                row["y"],
                row["tl_id"],
                fontsize=6,
                ha="center",
                va="bottom",
                zorder=5,
            )

    ax.set_title("Representative Snapshot: Node Influence Breadth")
    _style_map_axes(ax, positions, road_segments)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(out_path: Path, summary: dict[str, Any]) -> None:
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_node_influence_csv(
    out_path: Path,
    node_rows: list[dict[str, Any]],
    max_hops: int,
) -> None:
    hop_fields = [f"hop_{hop}" for hop in range(max_hops + 1)]
    fieldnames = [
        "node_index",
        "tl_id",
        "x",
        "y",
        "has_position",
        "is_sampled",
        "receptive_field_breadth",
        *hop_fields,
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in node_rows:
            csv_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(csv_row)


def compute_average_total_influence(
    model: nn.Module,
    snapshots: list[SnapshotGraph],
    *,
    max_hops: int,
    num_samples: int,
    device: torch.device | str,
) -> tuple[Tensor, float, list[Tensor], list[float]]:
    curves: list[Tensor] = []
    breadths: list[float] = []

    for snapshot in snapshots:
        try:
            curve, breadth = _total_influence_with_edge_attr(
                model,
                snapshot.data,
                max_hops=max_hops,
                num_samples=num_samples,
                normalize=True,
                average=True,
                device=device,
                seed=snapshot.step_index,
                vectorize=True,
            )
        except Exception as exc:  # pragma: no cover - exercised in integration
            _raise_with_pyg_hint(exc)

        curves.append(curve.detach().cpu())
        breadths.append(float(breadth))

    avg_curve = torch.stack(curves).mean(dim=0)
    avg_breadth = float(sum(breadths) / len(breadths))
    return avg_curve, avg_breadth, curves, breadths


def compute_node_influence_rows(
    model: nn.Module,
    snapshot: SnapshotGraph,
    positions: Tensor,
    tl_ids: list[str],
    *,
    max_hops: int,
    num_samples: int,
    device: torch.device | str,
) -> list[dict[str, Any]]:
    sampled_nodes = set(select_sampled_nodes(snapshot.data.num_nodes, num_samples, seed=snapshot.step_index))
    node_rows: list[dict[str, Any]] = []

    for node_idx, tl_id in enumerate(tl_ids):
        has_position = bool(torch.isfinite(positions[node_idx]).all())
        row: dict[str, Any] = {
            "node_index": node_idx,
            "tl_id": tl_id,
            "x": float(positions[node_idx, 0].item()) if has_position else "",
            "y": float(positions[node_idx, 1].item()) if has_position else "",
            "has_position": has_position,
            "is_sampled": node_idx in sampled_nodes,
            "receptive_field_breadth": "",
        }

        for hop in range(max_hops + 1):
            row[f"hop_{hop}"] = ""

        if node_idx in sampled_nodes:
            try:
                influence_per_hop = _jacobian_l1_agg_per_hop_safe(
                    model,
                    snapshot.data,
                    max_hops=max_hops,
                    node_idx=node_idx,
                    device=device,
                    vectorize=True,
                ).detach().cpu()
            except Exception as exc:  # pragma: no cover - exercised in integration
                _raise_with_pyg_hint(exc)

            row["receptive_field_breadth"] = receptive_field_breadth(influence_per_hop)
            for hop, value in enumerate(influence_per_hop.tolist()):
                row[f"hop_{hop}"] = float(value)

        node_rows.append(row)

    return node_rows


def run_visualization(
    *,
    checkpoint_path: str | Path,
    env_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    out_dir: str | Path,
    device: str = "cpu",
    num_snapshots: int = 5,
    max_hops: int | None = None,
    curve_num_samples: int | None = None,
    map_num_samples: int | None = None,
) -> dict[str, Any]:
    """Run graph/influence visualization and save all artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)
    checkpoint_path = Path(checkpoint_path)

    env = TrafficSignalEnv(**env_cfg)
    try:
        td0 = env.reset()
        tl_ids = list(env.tl_ids)
        positions = env.graph_builder.node_positions.detach().cpu()  # type: ignore[union-attr]
        road_segments = extract_road_segments(env.graph_builder.net)  # type: ignore[union-attr]
        edge_index = td0["edge_index"].detach().cpu()
        edge_attr = _extract_edge_attr(td0)

        agent = load_agent_for_visualization(
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            td0=td0,
            num_actions=env.num_actions,
            device=device_obj,
        )
        influence_model = EncoderInfluenceModel(agent.encoder, edge_attr=edge_attr).to(device_obj)
        influence_model.eval()

        resolved_max_hops = resolve_max_hops(influence_model, max_hops)
        episode_steps = count_episode_steps(env, agent, device_obj)
        snapshot_steps = evenly_spaced_indices(episode_steps, num_snapshots)
        snapshots = collect_snapshot_graphs(env, agent, device_obj, snapshot_steps)
    finally:
        env.close()

    num_nodes = len(tl_ids)
    resolved_curve_samples = resolve_curve_num_samples(num_nodes, curve_num_samples)
    resolved_map_samples = resolve_map_num_samples(num_nodes, map_num_samples)

    avg_curve, avg_breadth, snapshot_curves, snapshot_breadths = compute_average_total_influence(
        influence_model,
        snapshots,
        max_hops=resolved_max_hops,
        num_samples=resolved_curve_samples,
        device=device_obj,
    )

    representative_snapshot = snapshots[len(snapshots) // 2]
    node_rows = compute_node_influence_rows(
        influence_model,
        representative_snapshot,
        positions,
        tl_ids,
        max_hops=resolved_max_hops,
        num_samples=resolved_map_samples,
        device=device_obj,
    )

    plot_graph_topology(
        out_dir / "graph_topology.png",
        positions,
        edge_index,
        tl_ids,
        road_segments=road_segments,
    )
    plot_influence_curve(
        out_dir / "influence_curve.png",
        avg_curve,
        snapshot_curves,
        avg_breadth,
    )
    plot_node_influence_map(
        out_dir / "influence_map.png",
        positions,
        edge_index,
        tl_ids,
        node_rows,
        road_segments=road_segments,
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "net_file": str(env_cfg["net_file"]),
        "route_file": str(env_cfg["route_file"]),
        "device": str(device_obj),
        "num_nodes": len(tl_ids),
        "num_directed_edges": int(edge_index.shape[1]),
        "num_undirected_edges": len(deduplicate_undirected_edges(edge_index)),
        "num_road_segments": len(road_segments),
        "episode_steps": episode_steps,
        "snapshot_steps": snapshot_steps,
        "representative_snapshot_step": representative_snapshot.step_index,
        "max_hops": resolved_max_hops,
        "curve_num_samples": resolved_curve_samples,
        "map_num_samples": resolved_map_samples,
        "avg_total_influence": [float(v) for v in avg_curve.tolist()],
        "avg_receptive_field_breadth": avg_breadth,
        "snapshot_receptive_field_breadths": snapshot_breadths,
        "snapshot_total_influence": [
            [float(v) for v in curve.tolist()]
            for curve in snapshot_curves
        ],
        "artifacts": {
            "graph_topology": str(out_dir / "graph_topology.png"),
            "influence_curve": str(out_dir / "influence_curve.png"),
            "influence_map": str(out_dir / "influence_map.png"),
            "summary_json": str(out_dir / "influence_summary.json"),
            "node_csv": str(out_dir / "node_influence.csv"),
        },
    }

    write_summary_json(out_dir / "influence_summary.json", summary)
    write_node_influence_csv(out_dir / "node_influence.csv", node_rows, resolved_max_hops)
    return summary
