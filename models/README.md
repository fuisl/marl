# Models Overview

This folder contains the model stack used by the current MARL baseline.

## Current Baseline

The active architecture is `LocalNeighborGATDiscreteSAC` from
[models/local_neighbor_gat_discrete_sac.py](models/local_neighbor_gat_discrete_sac.py).

High-level flow:

1. Local encoder (`LocalEncoder`): MLP over each agent's own observation.
2. Neighbor encoder (`NeighborEncoder`): GATv2 over graph connectivity.
3. Fusion (`FusionMLP`): combine local + neighbor latents.
4. Actor (`SharedDiscreteActor`): output discrete action logits.
5. Critic (`CentralizedTwinCritic`): twin Q networks with pooled context.
6. Target critic: Polyak-updated copy for SAC stability.

## File-by-File Summary

- [models/local_neighbor_gat_discrete_sac.py](models/local_neighbor_gat_discrete_sac.py)
  - Main baseline agent class and encoder/fusion blocks.
  - Exposes `encode`, `select_action`, actor/critic query methods, `soft_update_target`.

- [models/actor.py](models/actor.py)
  - Shared discrete policy head (`SharedDiscreteActor`).
  - Supports action masking and sampling/argmax selection.

- [models/critic.py](models/critic.py)
  - Centralized twin Q critics (`CentralizedTwinCritic`).
  - Uses mean-pooled context over agents/nodes for CTDE.

- [models/graph_encoder.py](models/graph_encoder.py)
  - Generic graph encoder kept for compatibility and experiments.
  - The current baseline does not instantiate it directly.

- [models/__init__.py](models/__init__.py)
  - Public exports used by imports elsewhere in repo.

## Inputs and Shapes (Baseline)

- Observation tensor: `[N, obs_dim]` (or batched `[B, N, obs_dim]` in losses).
- Graph connectivity:
  - `edge_index`: `[2, E]`
  - `edge_attr`: `[E, edge_dim]` (optional)
- Agent-node mapping for all-intersections mode:
  - `agent_node_indices`, `agent_node_mask` for pooling node latents back to RL agents.

## Action Space

The actor outputs a categorical distribution over discrete actions per signal.

- Global action dimension is padded to max actions across signals.
- Environment-provided `action_mask` disables invalid actions.

## Training Hooks

The model is trained by:

- [train/discrete_sac_loop.py](train/discrete_sac_loop.py)
- [rl/losses.py](rl/losses.py)

with replay-based Discrete SAC and adaptive entropy temperature (`alpha`).

## Notes

- Main config aliases:
  - `algo=sac`
  - `model=gat`
- Under the hood, these aliases resolve to the Local-Neighbor GAT Discrete SAC baseline.
