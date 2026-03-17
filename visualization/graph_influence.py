"""Graph topology and City-Networks-style influence visualization helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

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


def format_node_label(node_id: str, attached_rl_ids: tuple[str, ...] | list[str]) -> str:
    if not attached_rl_ids:
        return node_id
    return f"{node_id}\n[{', '.join(attached_rl_ids)}]"


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
            td.get("graph_observation", td["agents", "observation"]),
            td["edge_index"],
            td.get("edge_attr", None),
            td["agents", "action_mask"],
            deterministic=True,
            agent_node_indices=td["agent_node_indices"],
            agent_node_mask=td["agent_node_mask"],
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
            graph_obs = td.get("graph_observation", td["agents", "observation"])
            snapshots.append(
                SnapshotGraph(
                    step_index=step_index,
                    data=Data(
                        x=graph_obs.detach().cpu(),
                        edge_index=td["edge_index"].detach().cpu(),
                        edge_attr=_extract_edge_attr(td),
                        num_nodes=int(graph_obs.shape[0]),
                    ),
                )
            )

        actions, _ = agent.select_action(
            td.get("graph_observation", td["agents", "observation"]),
            td["edge_index"],
            td.get("edge_attr", None),
            td["agents", "action_mask"],
            deterministic=True,
            agent_node_indices=td["agent_node_indices"],
            agent_node_mask=td["agent_node_mask"],
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
    obs_dim = int(td0.get("graph_observation", td0["agents", "observation"]).shape[-1])

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
    node_labels: list[str],
    node_is_signal: list[bool] | None = None,
    *,
    method_name: str,
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
    if node_is_signal is None or len(node_is_signal) != len(node_labels):
        node_is_signal = [False] * len(node_labels)

    signal_mask = torch.tensor(node_is_signal, dtype=torch.bool) & valid
    non_signal_mask = (~torch.tensor(node_is_signal, dtype=torch.bool)) & valid

    regular_size = 68 if len(node_labels) <= 100 else 14
    signal_size = 94 if len(node_labels) <= 100 else 24

    ax.scatter(
        positions[non_signal_mask, 0].tolist(),
        positions[non_signal_mask, 1].tolist(),
        s=regular_size,
        c="#0057A8",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label="Intersection node",
    )
    if bool(signal_mask.any()):
        ax.scatter(
            positions[signal_mask, 0].tolist(),
            positions[signal_mask, 1].tolist(),
            s=signal_size,
            c="#E86A33",
            marker="D",
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
            label="Traffic control signal",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    if len(node_labels) <= 100:
        for idx, node_label in enumerate(node_labels):
            if not bool(valid[idx]):
                continue
            ax.text(
                float(positions[idx, 0]),
                float(positions[idx, 1]),
                node_label,
                fontsize=6,
                ha="center",
                va="bottom",
                zorder=3,
            )

    ax.set_title(
        f"Intersection Graph Topology [{method_name}] ({len(node_labels)} nodes, {len(edges)} undirected edges)"
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
    node_labels: list[str],
    node_rows: list[dict[str, Any]],
    *,
    method_name: str,
    source_node_index: int | None,
    source_node_label: str | None,
    hop_influence_curve: Tensor | None,
    show_blue_edges: bool = True,
    show_heat_contours: bool = True,
    num_heat_contours: int = 6,
    heat_grid_size: int = 220,
    heat_sigma_scale: float = 0.06,
    heat_percentile_low: float = 2.0,
    heat_percentile_high: float = 98.0,
    heat_weight_mode: str = "raw",
    road_segments: list[RoadSegment] | None = None,
) -> None:
    def _robust_log_normalize(values: np.ndarray) -> np.ndarray:
        safe = np.maximum(values.astype(np.float64), 0.0)
        logged = np.log1p(safe)
        lo = float(np.percentile(logged, heat_percentile_low))
        hi = float(np.percentile(logged, heat_percentile_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(logged.min())
            hi = float(logged.max())
        if hi <= lo:
            return np.zeros_like(logged)
        clipped = np.clip(logged, lo, hi)
        return (clipped - lo) / (hi - lo)

    def _compute_smoothed_field(
        xs: np.ndarray,
        ys: np.ndarray,
        weights: np.ndarray,
        bounds: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_min, x_max, y_min, y_max = bounds
        grid_n = max(80, int(heat_grid_size))
        gx = np.linspace(x_min, x_max, grid_n)
        gy = np.linspace(y_min, y_max, grid_n)
        grid_x, grid_y = np.meshgrid(gx, gy)

        diag = max(np.hypot(x_max - x_min, y_max - y_min), 1.0)
        sigma = max(diag * float(heat_sigma_scale), 1e-3)
        sigma2 = sigma * sigma

        field = np.zeros_like(grid_x, dtype=np.float64)
        weight_sum = float(np.maximum(weights.sum(), 1e-12))
        for px, py, weight in zip(xs, ys, weights, strict=True):
            if weight <= 0.0:
                continue
            dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
            field += float(weight) * np.exp(-0.5 * dist2 / sigma2)

        return grid_x, grid_y, field / weight_sum

    def _draw_base_graph(ax: plt.Axes, edges: list[tuple[int, int]]) -> None:
        _plot_road_segments_with_style(
            ax,
            road_segments or [],
            color="#D2D7DE",
            alpha=0.62,
            base_linewidth=0.28,
            linewidth_scale=0.85,
            zorder=0,
        )
        if show_blue_edges:
            _plot_edges(ax, positions, edges, color="#5D8DBA", linewidth=0.7, alpha=0.35)

    def _draw_light_roads_foreground(ax: plt.Axes) -> None:
        _plot_road_segments_with_style(
            ax,
            road_segments or [],
            color="#C9D0D8",
            alpha=0.38,
            base_linewidth=0.22,
            linewidth_scale=0.65,
            zorder=3,
        )

    edges = deduplicate_undirected_edges(edge_index)
    is_focal_mode = (
        source_node_index is not None
        and source_node_label is not None
        and hop_influence_curve is not None
    )
    if is_focal_mode:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(21, 7.2),
            gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]},
        )
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14.5, 7.2),
            gridspec_kw={"width_ratios": [1.0, 1.0]},
        )

    valid_rows = [row for row in node_rows if row["has_position"]]
    if not valid_rows:
        raise RuntimeError("No finite node positions available for influence map.")

    xs = np.asarray([row["x"] for row in valid_rows], dtype=np.float64)
    ys = np.asarray([row["y"] for row in valid_rows], dtype=np.float64)
    node_influence_raw = np.asarray([row["influence_raw"] for row in valid_rows], dtype=np.float64)
    node_influence_norm = np.asarray([row["influence_normalized"] for row in valid_rows], dtype=np.float64)
    color_vals = _robust_log_normalize(node_influence_raw)

    if heat_weight_mode == "raw":
        heat_weights = np.maximum(node_influence_raw, 0.0)
    elif heat_weight_mode == "normalized":
        heat_weights = np.maximum(node_influence_norm, 0.0)
    elif heat_weight_mode == "log_normalized":
        heat_weights = np.maximum(color_vals, 0.0)
    else:
        raise ValueError(
            f"Invalid heat_weight_mode={heat_weight_mode!r}. Use one of: raw, normalized, log_normalized."
        )
    is_signal = np.asarray([bool(row.get("attached_rl_ids", "")) for row in valid_rows], dtype=bool)

    focal_x = None
    focal_y = None
    if source_node_index is not None:
        focal_x = float(positions[source_node_index, 0].item())
        focal_y = float(positions[source_node_index, 1].item())
    bounds = _compute_plot_bounds(positions, road_segments)
    if bounds is None:
        raise RuntimeError("Unable to determine plot bounds for influence map.")

    raw_ax = axes[0]
    raw_ax.set_facecolor("#FAFBFC")
    _draw_base_graph(raw_ax, edges)
    raw_ax.scatter(xs, ys, s=6, c="#BBC4CE", edgecolors="none", alpha=0.35, zorder=1)
    regular_mask = ~is_signal
    signal_mask = is_signal
    raw_scatter = raw_ax.scatter(
        xs[regular_mask],
        ys[regular_mask],
        s=24 if not is_focal_mode else 18,
        c=color_vals[regular_mask],
        cmap="magma",
        marker="o",
        edgecolors="none",
        alpha=0.72,
        zorder=2,
    )
    if bool(signal_mask.any()):
        raw_ax.scatter(
            xs[signal_mask],
            ys[signal_mask],
            s=28 if not is_focal_mode else 22,
            c=color_vals[signal_mask],
            cmap="magma",
            marker="D",
            edgecolors="none",
            alpha=0.76,
            zorder=2,
        )
    if focal_x is not None and focal_y is not None:
        raw_ax.scatter(
            [focal_x],
            [focal_y],
            s=170,
            c="#103B73",
            marker="*",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
    raw_ax.set_title("Raw Node Influence $I(v,u)$" if is_focal_mode else "Global Node Influence")
    _style_map_axes(raw_ax, positions, road_segments)
    raw_cbar = fig.colorbar(raw_scatter, ax=raw_ax, fraction=0.046, pad=0.02)
    raw_cbar.set_label("Robust log-normalized influence")

    heat_ax = axes[1]
    heat_ax.set_facecolor("#FAFBFC")
    _draw_base_graph(heat_ax, edges)
    grid_x, grid_y, field = _compute_smoothed_field(xs, ys, heat_weights, bounds)
    im = heat_ax.imshow(
        field,
        origin="lower",
        extent=(bounds[0], bounds[1], bounds[2], bounds[3]),
        cmap="magma",
        alpha=0.8,
        zorder=1,
        aspect="auto",
    )
    if show_heat_contours and num_heat_contours > 0 and float(field.max()) > 0.0:
        levels = np.linspace(float(field.min()), float(field.max()), int(num_heat_contours) + 2)[1:-1]
        if levels.size > 0:
            heat_ax.contour(
                grid_x,
                grid_y,
                field,
                levels=levels,
                colors="black",
                linewidths=0.55,
                alpha=0.55,
                zorder=2,
            )
    _draw_light_roads_foreground(heat_ax)
    heat_ax.scatter(
        xs[regular_mask],
        ys[regular_mask],
        s=3,
        c="#AEB8C2",
        marker="o",
        edgecolors="none",
        alpha=0.32,
        zorder=4,
    )
    if bool(signal_mask.any()):
        heat_ax.scatter(
            xs[signal_mask],
            ys[signal_mask],
            s=7,
            c="#C8CDD3",
            marker="D",
            edgecolors="none",
            alpha=0.45,
            zorder=4,
        )
    if focal_x is not None and focal_y is not None:
        heat_ax.scatter(
            [focal_x],
            [focal_y],
            s=170,
            c="#103B73",
            marker="*",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
    heat_ax.set_title("Smoothed Spatial Influence Field")
    _style_map_axes(heat_ax, positions, road_segments)
    heat_cbar = fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.02)
    heat_cbar.set_label(
        "Smoothed influence density"
        if heat_weight_mode == "raw"
        else f"Smoothed influence density ({heat_weight_mode})"
    )

    if is_focal_mode and hop_influence_curve is not None and source_node_label is not None:
        decay_ax = axes[2]
        decay_ax.set_facecolor("white")
        hops = np.arange(int(hop_influence_curve.numel()))
        hop_values = hop_influence_curve.detach().cpu().numpy().astype(np.float64)
        baseline = float(hop_values[0]) if hop_values.size > 0 else 0.0
        if baseline > 0.0:
            hop_values = hop_values / baseline
        decay_ax.plot(hops.tolist(), hop_values.tolist(), color="#0F4C81", linewidth=2.2, marker="o")
        decay_ax.fill_between(hops.tolist(), hop_values.tolist(), color="#87AFC7", alpha=0.25)
        decay_ax.set_xlabel("Hop distance $h$")
        decay_ax.set_ylabel("$T_h(v) / T_0(v)$")
        decay_ax.set_title("Hop-Wise Influence Decay")
        decay_ax.grid(alpha=0.25, linewidth=0.6)
        source_short = source_node_label.split("\n", maxsplit=1)[0]
        decay_ax.text(
            0.03,
            0.97,
            f"Source: {source_short}\nR = {receptive_field_breadth(hop_influence_curve):.3f}",
            transform=decay_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#D4D8DD", "alpha": 0.9},
        )
        fig.suptitle(
            f"Focal-Node Jacobian Influence [{method_name}] | Source = {source_short}",
            y=1.02,
            fontsize=13,
        )
    else:
        fig.suptitle(
            f"Global Jacobian Influence Field [{method_name}]",
            y=1.02,
            fontsize=13,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(out_path: Path, summary: dict[str, Any]) -> None:
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_node_influence_csv(
    out_path: Path,
    node_rows: list[dict[str, Any]],
    max_hops: int,
) -> None:
    fieldnames = [
        "node_index",
        "node_id",
        "attached_rl_ids",
        "x",
        "y",
        "has_position",
        "is_source",
        "hop_distance_from_source",
        "influence_raw",
        "influence_normalized",
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


def _select_focal_node_index(
    positions: Tensor,
    attached_rl_ids_by_node: list[tuple[str, ...]],
    explicit_index: int | None,
) -> int:
    if explicit_index is None:
        raise ValueError("Focal node is not set.")
    return int(explicit_index)


def _shortest_hop_distances(
    edge_index: Tensor,
    num_nodes: int,
    source_node: int,
) -> list[int]:
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        adjacency[src_i].add(dst_i)
        adjacency[dst_i].add(src_i)

    distances = [-1] * num_nodes
    distances[source_node] = 0
    queue = [source_node]
    head = 0

    while head < len(queue):
        current = queue[head]
        head += 1
        for neighbor in adjacency[current]:
            if distances[neighbor] != -1:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    return distances


def compute_focal_influence_rows(
    model: nn.Module,
    snapshot: SnapshotGraph,
    positions: Tensor,
    node_ids: list[str],
    attached_rl_ids_by_node: list[tuple[str, ...]],
    *,
    max_hops: int,
    focal_node_index: int,
    device: torch.device | str,
) -> tuple[list[dict[str, Any]], Tensor]:
    try:
        influence = _jacobian_l1_safe(
            model,
            snapshot.data,
            max_hops=max_hops,
            node_idx=focal_node_index,
            device=device,
            vectorize=True,
        ).detach().cpu()
    except Exception as exc:  # pragma: no cover - exercised in integration
        _raise_with_pyg_hint(exc)

    hop_subsets = k_hop_subsets_exact(
        focal_node_index,
        max_hops,
        snapshot.data.edge_index,
        int(snapshot.data.num_nodes),
        influence.device,
    )
    hop_influence_curve = torch.stack([influence[subset].sum() for subset in hop_subsets]).detach().cpu()

    self_influence = float(influence[focal_node_index].item())
    normalizer = self_influence if self_influence > 0.0 else float(influence.max().item())
    if normalizer <= 0.0:
        normalizer = 1.0

    hop_distances = _shortest_hop_distances(snapshot.data.edge_index, snapshot.data.num_nodes, focal_node_index)
    node_rows: list[dict[str, Any]] = []

    for node_idx, node_id in enumerate(node_ids):
        attached_rl_ids = attached_rl_ids_by_node[node_idx]
        has_position = bool(torch.isfinite(positions[node_idx]).all())
        raw_influence = float(influence[node_idx].item())
        row: dict[str, Any] = {
            "node_index": node_idx,
            "node_id": node_id,
            "attached_rl_ids": ",".join(attached_rl_ids),
            "node_label": format_node_label(node_id, attached_rl_ids),
            "x": float(positions[node_idx, 0].item()) if has_position else "",
            "y": float(positions[node_idx, 1].item()) if has_position else "",
            "has_position": has_position,
            "is_source": node_idx == focal_node_index,
            "hop_distance_from_source": hop_distances[node_idx],
            "influence_raw": raw_influence,
            "influence_normalized": raw_influence / normalizer,
        }

        node_rows.append(row)

    return node_rows, hop_influence_curve


def compute_global_influence_rows(
    model: nn.Module,
    snapshot: SnapshotGraph,
    positions: Tensor,
    node_ids: list[str],
    attached_rl_ids_by_node: list[tuple[str, ...]],
    *,
    max_hops: int,
    device: torch.device | str,
) -> list[dict[str, Any]]:
    num_nodes = int(snapshot.data.num_nodes)
    aggregate = torch.zeros(num_nodes, dtype=torch.float32)

    for focal_idx in range(num_nodes):
        try:
            influence = _jacobian_l1_safe(
                model,
                snapshot.data,
                max_hops=max_hops,
                node_idx=focal_idx,
                device=device,
                vectorize=True,
            ).detach().cpu()
        except Exception as exc:  # pragma: no cover - exercised in integration
            _raise_with_pyg_hint(exc)
        aggregate += influence.to(dtype=torch.float32)

    aggregate = aggregate / max(float(num_nodes), 1.0)
    max_value = float(aggregate.max().item()) if num_nodes > 0 else 0.0
    if max_value <= 0.0:
        max_value = 1.0

    node_rows: list[dict[str, Any]] = []
    for node_idx, node_id in enumerate(node_ids):
        attached_rl_ids = attached_rl_ids_by_node[node_idx]
        has_position = bool(torch.isfinite(positions[node_idx]).all())
        raw_influence = float(aggregate[node_idx].item())
        node_rows.append(
            {
                "node_index": node_idx,
                "node_id": node_id,
                "attached_rl_ids": ",".join(attached_rl_ids),
                "node_label": format_node_label(node_id, attached_rl_ids),
                "x": float(positions[node_idx, 0].item()) if has_position else "",
                "y": float(positions[node_idx, 1].item()) if has_position else "",
                "has_position": has_position,
                "is_source": False,
                "hop_distance_from_source": -1,
                "influence_raw": raw_influence,
                "influence_normalized": raw_influence / max_value,
            }
        )

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
    show_blue_edges_influence_map: bool = True,
    focal_node_index: int | None = None,
    show_heat_contours: bool = True,
    num_heat_contours: int = 6,
    heat_grid_size: int = 220,
    heat_sigma_scale: float = 0.06,
    heat_percentile_low: float = 2.0,
    heat_percentile_high: float = 98.0,
    heat_weight_mode: str = "raw",
) -> dict[str, Any]:
    """Run graph/influence visualization and save all artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)
    checkpoint_path = Path(checkpoint_path)

    env = TrafficSignalEnv(**env_cfg)
    try:
        td0 = env.reset()
        node_ids = list(env.graph_builder.node_ids)  # type: ignore[union-attr]
        attached_rl_ids_by_node = list(env.graph_builder.attached_rl_ids_by_node)  # type: ignore[union-attr]
        node_labels = [
            format_node_label(node_id, attached_rl_ids)
            for node_id, attached_rl_ids in zip(
                node_ids,
                attached_rl_ids_by_node,
                strict=True,
            )
        ]
        node_is_signal = [len(attached_rl_ids) > 0 for attached_rl_ids in attached_rl_ids_by_node]
        positions = env.graph_builder.node_positions.detach().cpu()  # type: ignore[union-attr]
        road_segments = extract_road_segments(env.graph_builder.net)  # type: ignore[union-attr]
        edge_index = td0["edge_index"].detach().cpu()
        edge_attr = _extract_edge_attr(td0)
        graph_builder_mode = env.graph_builder.mode  # type: ignore[union-attr]

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

    num_nodes = len(node_ids)
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
    global_mode = focal_node_index is None

    resolved_focal_node_index: int | None = None
    focal_hop_curve: Tensor | None = None

    if global_mode:
        node_rows = compute_global_influence_rows(
            influence_model,
            representative_snapshot,
            positions,
            node_ids,
            attached_rl_ids_by_node,
            max_hops=resolved_max_hops,
            device=device_obj,
        )
    else:
        resolved_focal_node_index = _select_focal_node_index(
            positions,
            attached_rl_ids_by_node,
            focal_node_index,
        )
        node_rows, focal_hop_curve = compute_focal_influence_rows(
            influence_model,
            representative_snapshot,
            positions,
            node_ids,
            attached_rl_ids_by_node,
            max_hops=resolved_max_hops,
            focal_node_index=resolved_focal_node_index,
            device=device_obj,
        )

    plot_graph_topology(
        out_dir / "graph_topology.png",
        positions,
        edge_index,
        node_labels,
        node_is_signal,
        method_name=graph_builder_mode,
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
        node_labels,
        node_rows,
        method_name=graph_builder_mode,
        source_node_index=resolved_focal_node_index,
        source_node_label=(
            node_labels[resolved_focal_node_index]
            if resolved_focal_node_index is not None
            else None
        ),
        hop_influence_curve=focal_hop_curve,
        show_blue_edges=show_blue_edges_influence_map,
        show_heat_contours=show_heat_contours,
        num_heat_contours=num_heat_contours,
        heat_grid_size=heat_grid_size,
        heat_sigma_scale=heat_sigma_scale,
        heat_percentile_low=heat_percentile_low,
        heat_percentile_high=heat_percentile_high,
        heat_weight_mode=heat_weight_mode,
        road_segments=road_segments,
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "net_file": str(env_cfg["net_file"]),
        "route_file": str(env_cfg["route_file"]),
        "graph_builder_mode": graph_builder_mode,
        "device": str(device_obj),
        "num_nodes": len(node_ids),
        "num_directed_edges": int(edge_index.shape[1]),
        "num_undirected_edges": len(deduplicate_undirected_edges(edge_index)),
        "num_road_segments": len(road_segments),
        "episode_steps": episode_steps,
        "snapshot_steps": snapshot_steps,
        "representative_snapshot_step": representative_snapshot.step_index,
        "max_hops": resolved_max_hops,
        "curve_num_samples": resolved_curve_samples,
        "map_num_samples": resolved_map_samples,
        "show_blue_edges_influence_map": bool(show_blue_edges_influence_map),
        "heat_weight_mode": heat_weight_mode,
        "global_spatial_mode": bool(global_mode),
        "focal_node_index": resolved_focal_node_index,
        "focal_node_id": (
            node_ids[resolved_focal_node_index]
            if resolved_focal_node_index is not None
            else None
        ),
        "focal_hop_influence": (
            [float(v) for v in focal_hop_curve.tolist()]
            if focal_hop_curve is not None
            else None
        ),
        "focal_receptive_field_breadth": (
            receptive_field_breadth(focal_hop_curve)
            if focal_hop_curve is not None
            else None
        ),
        "avg_total_influence": [float(v) for v in avg_curve.tolist()],
        "avg_receptive_field_breadth": avg_breadth,
        "snapshot_receptive_field_breadths": snapshot_breadths,
        "snapshot_total_influence": [
            [float(v) for v in curve.tolist()]
            for curve in snapshot_curves
        ],
        "node_to_rl_ids": {
            node_id: list(attached_rl_ids)
            for node_id, attached_rl_ids in zip(
                node_ids,
                attached_rl_ids_by_node,
                strict=True,
            )
        },
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
