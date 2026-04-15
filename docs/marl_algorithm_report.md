# MARL Training/Architecture Report

This report summarizes how the current training algorithm and architecture work end-to-end.

## 1) Why both `algo` and `model` exist

Hydra separates concerns:

- `algo`: training algorithm and control flow (SAC, fixed, maxpressure, etc.).
- `model`: neural architecture used by that algorithm.

This makes it easy to swap either independently.

Current simplified aliases:

- `algo=sac` (Discrete SAC trainer)
- `model=gat` (Local-Neighbor GAT architecture)

These aliases map to the same baseline implementation already in use.

## 2) Current baseline stack

Main runtime path:

1. [scripts/run_experiment.py](scripts/run_experiment.py)
2. [train/discrete_sac_loop.py](train/discrete_sac_loop.py)
3. [marl_env/sumo_env.py](marl_env/sumo_env.py)
4. [models/local_neighbor_gat_discrete_sac.py](models/local_neighbor_gat_discrete_sac.py)
5. [rl/losses.py](rl/losses.py)

CTDE structure (centralized training, decentralized execution):

- Execution-time action per light uses local observation + neighbor context.
- Training-time critics use centralized context via pooled latent features.

## 3) Action space and masking

Action space is discrete per signal.

- Environment builds a max action dimension across all signals.
- Per-signal invalid actions are masked with `action_mask`.
- Actor logits for invalid actions are set to large negative values.

Policy distribution:

$$
\pi_\theta(a_i \mid s_i) = \text{Categorical}(\text{logits}_i)
$$

## 4) Observation and communication path

Canonical observations are created in [marl_env/sumo_env.py](marl_env/sumo_env.py) and transformed by [marl_env/observation_adapter.py](marl_env/observation_adapter.py).

Communication is graph-based:

- Nodes: intersections (including all-intersections graph mode).
- Edges: road connectivity from `GraphBuilder`.
- Message passing: GATv2 attention in neighbor encoder.

This is implicit communication through the graph neural network, not explicit message tensors between agents.

## 5) Model architecture (current)

In [models/local_neighbor_gat_discrete_sac.py](models/local_neighbor_gat_discrete_sac.py):

1. `LocalEncoder`: MLP on each agent's own features.
2. `NeighborEncoder`: GATv2 aggregation over neighbors.
3. `FusionMLP`: concatenate local and neighbor latents, then fuse.
4. `SharedDiscreteActor`: shared categorical policy head.
5. `CentralizedTwinCritic`: twin Q networks with pooled context.

If graph nodes outnumber RL-controlled agents, node latents are pooled back to agents via `agent_node_indices`/`agent_node_mask`.

## 6) Reward definitions and global reward

From [marl_env/reward.py](marl_env/reward.py):

- `wait`: per-agent reward = negative total waiting time.
- `pressure`: per-agent reward = negative queue pressure (incoming - outgoing).

Per-step global reward is the sum of per-agent rewards:

$$
R_t^{\text{global}} = \sum_{i=1}^{N} r_{i,t}
$$

Episode global reward reported in logs is accumulated over time:

$$
G^{\text{episode}} = \sum_{t=0}^{T-1} R_t^{\text{global}}
$$

## 7) Discrete SAC math used in implementation

Implementation is in [rl/losses.py](rl/losses.py).

Let critics output all-action Q values $Q_1(s, \cdot), Q_2(s, \cdot)$.

Target value:

$$
V_{\text{target}}(s') = \sum_a \pi(a\mid s')\Big(\min(Q_1',Q_2')(s',a) - \alpha \log \pi(a\mid s')\Big)
$$

Bellman target:

$$
y = r + \gamma (1-d) V_{\text{target}}(s')
$$

Critic loss:

$$
\mathcal{L}_Q = \frac{1}{2}\Big(\lVert Q_1(s,a)-y\rVert^2 + \lVert Q_2(s,a)-y\rVert^2\Big)
$$

Actor loss:

$$
\mathcal{L}_\pi = \mathbb{E}_{s}\left[\sum_a \pi(a\mid s)\big(\alpha\log\pi(a\mid s) - \min(Q_1,Q_2)(s,a)\big)\right]
$$

Temperature (entropy) tuning:

$$
\mathcal{L}_\alpha = \log\alpha \cdot (\mathcal{H}(\pi) - \mathcal{H}_{\text{target}})
$$

This is aligned with standard entropy-regularized actor-critic formulations used in references like Spinning Up style SAC discussions (adapted here for discrete actions and MARL graph context).

## 8) Training loop behavior (current)

[train/discrete_sac_loop.py](train/discrete_sac_loop.py) now uses clear stages:

1. Build env
2. Build observation/graph pipeline
3. Build baseline model (`LocalNeighborGATDiscreteSAC`)
4. Build optimizers
5. Build loss + replay
6. Run episodes and off-policy updates
7. Checkpoint best global reward
8. Optional postprocess/eval/visualization

## 9) Diagram status

Architecture and flow diagrams are documented in:

- [docs/ctde_graph_sac_architecture.md](docs/ctde_graph_sac_architecture.md)
- [docs/marl_env_diagrams.md](docs/marl_env_diagrams.md)

They show the Local-Neighbor GAT + Discrete SAC CTDE pipeline and the environment/graph data path.

## 10) What is related to MARL in this repo

Core MARL pieces:

- Environment and RESCO semantics:
  - [marl_env/sumo_env.py](marl_env/sumo_env.py)
  - [marl_env/reward.py](marl_env/reward.py)
  - [marl_env/observation_adapter.py](marl_env/observation_adapter.py)
  - [marl_env/action_constraints.py](marl_env/action_constraints.py)
  - [marl_env/graph_builder.py](marl_env/graph_builder.py)
- Models:
  - [models/local_neighbor_gat_discrete_sac.py](models/local_neighbor_gat_discrete_sac.py)
  - [models/actor.py](models/actor.py)
  - [models/critic.py](models/critic.py)
- RL logic:
  - [rl/losses.py](rl/losses.py)
  - [rl/replay.py](rl/replay.py)
  - [train/discrete_sac_loop.py](train/discrete_sac_loop.py)

## 11) Practical run commands (simplified names)

```bash
python scripts/run_experiment.py
python scripts/run_experiment.py scenario=grid5x5 algo=sac model=gat seed=21
python scripts/run_experiment.py -m scenario=grid4x4,grid5x5 seed=1,2
```

Evaluation/visualization:

```bash
python train/evaluate.py checkpoint_path=<PATH_TO_CHECKPOINT>
python visualization/graph_influence.py checkpoint_path=<PATH_TO_CHECKPOINT>
```
