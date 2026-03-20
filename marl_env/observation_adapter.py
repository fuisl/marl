"""Canonical observation layout and feature adapters for the unified env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from marl_env.resco_observation import RescoSignalState


FEATURE_MODES: tuple[str, ...] = ("snapshot", "wave", "mplight", "drq")


@dataclass(frozen=True)
class CanonicalObservationLayout:
    """Flat layout for the canonical per-signal snapshot."""

    max_lanes: int

    @property
    def feature_dim(self) -> int:
        return 2 + 5 * self.max_lanes

    @property
    def phase_index_slice(self) -> slice:
        return slice(0, 1)

    @property
    def phase_length_slice(self) -> slice:
        return slice(1, 2)

    @property
    def lane_mask_slice(self) -> slice:
        start = 2
        return slice(start, start + self.max_lanes)

    @property
    def approaching_slice(self) -> slice:
        start = self.lane_mask_slice.stop
        return slice(start, start + self.max_lanes)

    @property
    def queued_slice(self) -> slice:
        start = self.approaching_slice.stop
        return slice(start, start + self.max_lanes)

    @property
    def total_wait_slice(self) -> slice:
        start = self.queued_slice.stop
        return slice(start, start + self.max_lanes)

    @property
    def total_speed_slice(self) -> slice:
        start = self.total_wait_slice.stop
        return slice(start, start + self.max_lanes)

    def as_dict(self) -> dict[str, int]:
        return {
            "max_lanes": int(self.max_lanes),
            "feature_dim": int(self.feature_dim),
        }

    def split(self, observation: Tensor) -> dict[str, Tensor]:
        if observation.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected canonical snapshot width {self.feature_dim}, got {observation.shape[-1]}."
            )
        return {
            "phase_index": observation[..., self.phase_index_slice],
            "phase_length": observation[..., self.phase_length_slice],
            "lane_mask": observation[..., self.lane_mask_slice],
            "approaching": observation[..., self.approaching_slice],
            "queued": observation[..., self.queued_slice],
            "total_wait": observation[..., self.total_wait_slice],
            "total_speed": observation[..., self.total_speed_slice],
        }


def build_canonical_snapshot(
    *,
    signal: RescoSignalState,
    layout: CanonicalObservationLayout,
) -> Tensor:
    observation = signal.observation
    lane_mask = [1.0] * len(signal.lanes)
    approaching: list[float] = []
    queued: list[float] = []
    total_wait: list[float] = []
    total_speed: list[float] = []

    for lane_id in signal.lanes:
        lane_obs = observation.get_lane(lane_id)
        approaching.append(float(lane_obs.approaching))
        queued.append(float(lane_obs.queued))
        wait_sum = 0.0
        speed_sum = 0.0
        for vehicle in lane_obs.vehicles.values():
            wait_sum += float(vehicle.wait)
            speed_sum += float(vehicle.average_speed)
        total_wait.append(wait_sum)
        total_speed.append(speed_sum)

    pad = layout.max_lanes - len(signal.lanes)
    if pad < 0:
        raise ValueError(
            f"Signal {signal.signal_id!r} has {len(signal.lanes)} lanes, "
            f"which exceeds max_lanes={layout.max_lanes}."
        )
    if pad:
        zeros = [0.0] * pad
        lane_mask.extend(zeros)
        approaching.extend(zeros)
        queued.extend(zeros)
        total_wait.extend(zeros)
        total_speed.extend(zeros)

    values = [
        float(observation.current_phase),
        float(observation.phase_length),
        *lane_mask,
        *approaching,
        *queued,
        *total_wait,
        *total_speed,
    ]
    return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class GraphMetadata:
    edge_index: Tensor
    edge_attr: Tensor | None
    node_ids: tuple[str, ...]
    attached_rl_ids_by_node: tuple[tuple[str, ...], ...]
    agent_node_indices: Tensor
    agent_node_mask: Tensor


class ObservationAdapter:
    """Build algorithm-specific features from canonical env snapshots."""

    def __init__(
        self,
        *,
        signal_specs: dict[str, dict[str, Any]],
        tl_ids: list[str],
        layout: CanonicalObservationLayout,
        graph_metadata: GraphMetadata | None = None,
    ) -> None:
        self.signal_specs = signal_specs
        self.tl_ids = list(tl_ids)
        self.layout = layout
        self.graph_metadata = graph_metadata
        self._tl_index = {tl_id: idx for idx, tl_id in enumerate(self.tl_ids)}

    def as_state_dict(
        self,
        observations: Tensor,
        *,
        feature_mode: str,
    ) -> dict[str, list[float]]:
        features = self.agent_features(observations, feature_mode=feature_mode)
        if features.dim() != 2:
            raise ValueError("State dict conversion expects per-agent observations without a batch dimension.")
        return {
            tl_id: [float(v) for v in features[idx].tolist()]
            for idx, tl_id in enumerate(self.tl_ids)
        }

    def agent_features(
        self,
        observations: Tensor,
        *,
        feature_mode: str,
    ) -> Tensor:
        if feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"Unsupported feature mode {feature_mode!r}. Expected one of {FEATURE_MODES}."
            )
        if feature_mode == "snapshot":
            return observations.clone()

        squeezed = observations.dim() == 2
        if squeezed:
            observations = observations.unsqueeze(0)
        if observations.dim() != 3:
            raise ValueError(
                "ObservationAdapter expects tensors shaped [n_agents, obs_dim] or [batch, n_agents, obs_dim]."
            )

        batch_rows: list[Tensor] = []
        for batch_index in range(observations.shape[0]):
            per_agent_rows: list[Tensor] = []
            for agent_index, tl_id in enumerate(self.tl_ids):
                snapshot = observations[batch_index, agent_index]
                spec = self.signal_specs[tl_id]
                if feature_mode == "wave":
                    row = self._wave_row(snapshot, spec)
                elif feature_mode == "mplight":
                    row = self._mplight_row(snapshot, observations[batch_index], spec)
                else:
                    row = self._drq_row(snapshot, spec)
                per_agent_rows.append(row)
            batch_rows.append(torch.stack(per_agent_rows, dim=0))

        result = torch.stack(batch_rows, dim=0)
        return result[0] if squeezed else result

    def graph_features(
        self,
        observations: Tensor,
        *,
        feature_mode: str,
    ) -> Tensor:
        if self.graph_metadata is None:
            return self.agent_features(observations, feature_mode=feature_mode)

        agent_features = self.agent_features(observations, feature_mode=feature_mode)
        squeezed = agent_features.dim() == 2
        if squeezed:
            agent_features = agent_features.unsqueeze(0)

        feature_dim = int(agent_features.shape[-1])
        graph_rows: list[Tensor] = []
        for batch_index in range(agent_features.shape[0]):
            node_rows: list[Tensor] = []
            for attached_ids in self.graph_metadata.attached_rl_ids_by_node:
                indices = [
                    self._tl_index[tl_id]
                    for tl_id in attached_ids
                    if tl_id in self._tl_index
                ]
                if not indices:
                    node_rows.append(
                        torch.zeros(
                            feature_dim,
                            dtype=agent_features.dtype,
                            device=agent_features.device,
                        )
                    )
                    continue
                node_rows.append(agent_features[batch_index, indices].mean(dim=0))
            graph_rows.append(torch.stack(node_rows, dim=0))

        result = torch.stack(graph_rows, dim=0)
        return result[0] if squeezed else result

    def _wave_row(self, snapshot: Tensor, spec: dict[str, Any]) -> Tensor:
        fields = self.layout.split(snapshot)
        lane_indices = self._lane_indices(spec)
        values: list[float] = []
        for direction in spec["directions"]:
            total = 0.0
            for idx in lane_indices[direction]:
                total += float(fields["queued"][idx].item())
                total += float(fields["approaching"][idx].item())
            values.append(total)
        return torch.tensor(values, dtype=torch.float32, device=snapshot.device)

    def _mplight_row(
        self,
        snapshot: Tensor,
        all_snapshots: Tensor,
        spec: dict[str, Any],
    ) -> Tensor:
        fields = self.layout.split(snapshot)
        phase_index = float(fields["phase_index"].item())
        lane_indices = self._lane_indices(spec)

        values = [phase_index]
        for direction in spec["directions"]:
            pressure = 0.0
            for idx in lane_indices[direction]:
                pressure += float(fields["queued"][idx].item())
            for lane_id in spec.get("lane_sets_outbound", {}).get(direction, []):
                downstream_signal_id = spec.get("out_lane_to_signal_id", {}).get(lane_id)
                if downstream_signal_id not in self.signal_specs:
                    continue
                downstream_fields = self.layout.split(
                    all_snapshots[self._tl_index[downstream_signal_id]]
                )
                downstream_indices = self._lane_index_map(self.signal_specs[downstream_signal_id])
                if lane_id not in downstream_indices:
                    continue
                pressure -= float(downstream_fields["queued"][downstream_indices[lane_id]].item())
            values.append(pressure)
        return torch.tensor(values, dtype=torch.float32, device=snapshot.device)

    def _drq_row(self, snapshot: Tensor, spec: dict[str, Any]) -> Tensor:
        fields = self.layout.split(snapshot)
        current_phase = int(fields["phase_index"].item())
        lane_order = list(spec["lane_order"])
        values: list[float] = []
        for lane_idx, lane_id in enumerate(lane_order):
            lane_index_map = self._lane_index_map(spec)
            field_idx = lane_index_map[lane_id]
            values.extend(
                [
                    1.0 if lane_idx == current_phase else 0.0,
                    float(fields["approaching"][field_idx].item()),
                    float(fields["total_wait"][field_idx].item()),
                    float(fields["queued"][field_idx].item()),
                    float(fields["total_speed"][field_idx].item()),
                ]
            )
        return torch.tensor(values, dtype=torch.float32, device=snapshot.device)

    def _lane_index_map(self, spec: dict[str, Any]) -> dict[str, int]:
        return {
            lane_id: idx
            for idx, lane_id in enumerate(spec["lane_order"])
        }

    def _lane_indices(self, spec: dict[str, Any]) -> dict[str, list[int]]:
        lane_index_map = self._lane_index_map(spec)
        return {
            direction: [
                lane_index_map[lane_id]
                for lane_id in spec["lane_sets"][direction]
                if lane_id in lane_index_map
            ]
            for direction in spec["directions"]
        }


def build_graph_metadata(
    *,
    edge_index: Tensor,
    edge_attr: Tensor | None,
    node_ids: list[str],
    attached_rl_ids_by_node: list[tuple[str, ...]],
    agent_node_indices: Tensor,
    agent_node_mask: Tensor,
) -> GraphMetadata:
    return GraphMetadata(
        edge_index=edge_index.clone(),
        edge_attr=None if edge_attr is None else edge_attr.clone(),
        node_ids=tuple(node_ids),
        attached_rl_ids_by_node=tuple(tuple(ids) for ids in attached_rl_ids_by_node),
        agent_node_indices=agent_node_indices.clone(),
        agent_node_mask=agent_node_mask.clone(),
    )
