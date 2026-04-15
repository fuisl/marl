# Presentation Script: What This Repo Does

## 0. Opening

Today I will explain this repository end to end:
- what problem it solves,
- how one experiment is executed,
- how one episode is collected and learned from,
- how the MARL model works mathematically,
- and how the classical baselines work for comparison.

This project is a graph-based multi-agent reinforcement learning system for traffic signal control in SUMO.

## 1. What is the core problem?

We control many traffic lights at once.
Each intersection is an agent.
Agents must choose discrete signal actions repeatedly to reduce congestion.

The challenge is coordination:
- local decisions affect nearby intersections,
- traffic state is dynamic and partially local,
- and each signal can have different legal local actions.

## 2. Main entrypoint and configuration

When we run an experiment, we start from:
- scripts/run_experiment.py

That script uses Hydra configs from:
- configs/run.yaml
- configs/algo/
- configs/model/
- configs/scenario/
- configs/train/

Important separation:
- algo means training procedure and dispatcher logic.
- model means neural architecture.

Current simplified default aliases:
- algo=sac
- model=gat

Internally this resolves to:
- discrete SAC trainer
- Local-Neighbor GAT Discrete SAC architecture

## 3. Runtime dispatch: what gets fired

In scripts/run_experiment.py:
1. Configs are composed.
2. Env settings are merged.
3. algo.trainer is inspected.

If trainer is discrete_sac, it calls:
- train/discrete_sac_loop.py

If trainer is fixed_time_baseline, it calls:
- train/fixed_time_baseline.py

So this repo supports both learning-based and classical/static baselines via one runner.

## 4. Training pipeline in train/discrete_sac_loop.py

The training loop is staged clearly:
1. Build SUMO environment.
2. Build observation graph pipeline.
3. Build Local-Neighbor GAT SAC agent.
4. Build optimizers.
5. Build loss computer and replay buffer.
6. Run episodes.
7. After each episode, run SAC updates from replay.
8. Save best checkpoint by global reward.
9. Optional postprocessing and visualization hooks.

Key outputs:
- runs/.../train_log.csv
- runs/.../best_agent.pt

## 5. Observation graph pipeline: why it exists

Raw environment observations are per-agent snapshots.
The graph model needs node-aware graph features plus topology.

This bridge is implemented using:
- marl_env/observation_adapter.py
- env graph metadata from marl_env/graph_builder.py

So each step converts snapshots to graph features and feeds:
- node features,
- edge index,
- edge attributes,
- and mapping between graph nodes and RL-controlled agents.

Without this layer, the GAT model would not receive structured neighborhood context.

## 6. What happens in one episode

At each RL step:
1. Current observations are transformed to graph features.
2. Agent computes action distribution and samples actions.
3. Environment applies actions in SUMO.
4. Environment advances simulation for delta_t.
5. New observations and rewards are produced.
6. Transition is packed into replay buffer.

This repeats until done.
Then training samples mini-batches from replay and runs gradient updates.

## 7. Action space and masking

Action space is discrete per signal.
But different intersections may have different numbers of legal local actions.

The environment provides action_mask.
In models/actor.py, invalid actions are masked in logits by a very negative value.

Effect:
- policy never samples illegal actions,
- one shared actor can still handle heterogeneous intersections.

## 8. Reward design and global reward

Reward functions are defined in:
- marl_env/reward.py

Current built-in rewards include:
- wait reward: negative total waiting time
- pressure reward: negative queue pressure difference

Per step global reward is sum of all agents:

$R_t^{global} = \sum_{i=1}^{N} r_{i,t}$

Episode global reward accumulates over time:

$G = \sum_{t=0}^{T-1} R_t^{global}$

This is the main scalar used for tracking and best-checkpoint selection.

## 9. Model architecture in models/local_neighbor_gat_discrete_sac.py

The model has three representation stages:

1. LocalEncoder
- MLP on each agent local observation.

2. NeighborEncoder
- GATv2 graph attention over connectivity.
- captures neighboring influence.

3. FusionMLP
- concatenates local and neighbor latent vectors,
- produces final latent used by actor and critic.

Then heads:
- SharedDiscreteActor in models/actor.py
- CentralizedTwinCritic in models/critic.py

Target critic is Polyak-updated for stability.
Entropy temperature alpha is learned.

## 10. SAC mathematics used in rl/losses.py

For discrete SAC, critics output Q-values for all actions.

Target state value:

$V(s') = \sum_a \pi(a|s') [\min(Q_1',Q_2')(s',a) - \alpha \log \pi(a|s')]$

Bellman target:

$y = r + \gamma (1-d) V(s')$

Critic loss:

$L_Q = 0.5[(Q_1(s,a)-y)^2 + (Q_2(s,a)-y)^2]$

Actor loss:

$L_\pi = E_s[\sum_a \pi(a|s)(\alpha \log \pi(a|s) - \min(Q_1,Q_2)(s,a))]$

Temperature loss:

$L_\alpha = \log \alpha (H(\pi) - H_{target})$

This matches entropy-regularized actor-critic principles from modern SAC literature.

## 11. How classical baselines work

Classical/static methods are implemented in:
- train/fixed_time_baseline.py
- train/resco_baselines.py

Included strategies:
- FIXED timing controller
- STOCHASTIC random action baseline
- MAXWAVE heuristic
- MAXPRESSURE heuristic

These do not learn neural parameters.
They choose actions from hand-designed rules each step.

Why they matter:
- sanity checks,
- lower-bound and heuristic comparisons,
- helps show value added by learning.

## 12. What to say if asked why this approach

Reasoning:
- Traffic is a network process, not independent intersections.
- Pure local policy misses spillback and cross-intersection effects.
- Graph attention provides structured neighbor aggregation.
- Discrete SAC handles stochasticity and exploration with entropy regularization.

So this repo combines graph inductive bias with off-policy sample reuse.

## 13. Typical experiment commands

Train default:
- python scripts/run_experiment.py

Train explicit:
- python scripts/run_experiment.py scenario=grid5x5 algo=sac model=gat seed=21

Sweep:
- python scripts/run_experiment.py -m scenario=grid4x4,grid5x5 seed=1,2

Run classical baseline:
- python scripts/run_experiment.py algo=fixed

## 14. Closing

In one sentence:
This repository is a complete MARL traffic-control pipeline that runs SUMO, builds graph-structured observations, trains a Local-Neighbor GAT Discrete SAC agent with replay-based CTDE learning, and benchmarks against classical non-learning controllers.
