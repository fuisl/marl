# CTDE Graph-SAC Architecture In This Repository

This note explains the current implementation, not an abstract MARL template.

It is based on the code paths that are actually used today:

- `marl_env/sumo_env.py`
- `marl_env/graph_builder.py`
- `models/graph_encoder.py`
- `models/actor.py`
- `models/critic.py`
- `models/marl_discrete_sac.py`
- `rl/losses.py`
- `rl/rollout.py`
- `scripts/train_gat_baseline.py`

The shortest summary is:

1. SUMO produces traffic-light agents and a static road graph.
2. Every decision step, the environment builds node features.
3. A GATv2 encoder turns those node features into latent embeddings.
4. A shared actor chooses one discrete green-phase action per RL-controlled light.
5. A centralized twin critic evaluates those actions using both local agent latent and a pooled graph context.
6. Training is done with Discrete SAC and replay.

The implementation is CTDE, but with an important nuance:

- The critic is centralized during training.
- The actor head is per-agent and shared.
- The actor still consumes graph-encoded state built from the full graph observation that the simulator exposes.

So this is "centralized training, per-agent execution heads over a shared graph state", not strict local-information decentralization.

## 1. Start From One Intersection

Suppose there were only one controlled intersection.

Its raw observation would be a vector of lane statistics, plus phase state:

$$
x_i =
\big[
q_i,\;
w_i,\;
o_i,\;
s_i,\;
p_i,\;
\tau_i
\big]
$$

where:

- $q_i$: padded halting counts on lanes
- $w_i$: padded waiting times
- $o_i$: padded occupancies
- $s_i$: padded mean speeds
- $p_i$: one-hot encoding of the current controllable green phase
- $\tau_i$: elapsed time in the current green

In code, this comes from `TrafficSignalEnv._build_observation_from_lanes_and_phase(...)`.

For a lane set $\mathcal L(i)$, the padded traffic blocks are:

$$
q_i = \mathrm{pad}\big([\,\text{halting}(\ell)\,]_{\ell \in \mathcal L(i)}, L_{\max}\big)
$$
$$
w_i = \mathrm{pad}\big([\,\text{waiting}(\ell)\,]_{\ell \in \mathcal L(i)}, L_{\max}\big)
$$
$$
o_i = \mathrm{pad}\big([\,\text{occupancy}(\ell)\,]_{\ell \in \mathcal L(i)}, L_{\max}\big)
$$
$$
s_i = \mathrm{pad}\big([\,\text{speed}(\ell)\,]_{\ell \in \mathcal L(i)}, L_{\max}\big)
$$

If we stopped here, the policy would be a standard per-agent MLP:

$$
\pi_i(a \mid x_i)
$$

But this repository does not stop here, because traffic lights interact through the road network.

## 2. Add The Road Graph

Now replace the isolated intersection with a graph:

$$
G = (V, E)
$$

The exact meaning of \(V\) depends on `graph_builder_mode`.

### 2.1 Graph construction modes

The current implementation supports three modes in `marl_env/graph_builder.py`.

### `original`

- Nodes: only RL-controlled traffic-light IDs
- Edges: immediate adjacent controlled neighbors only

This reproduces the old behavior:

$$
V = \{\text{controlled TLS ids}\}
(i,j) \in E
\iff
j \text{ is the first immediate controlled neighbor seen from } i
$$

### `walk_to_light`

- Nodes: still only RL-controlled traffic-light IDs
- Edges: walk through unsignalized SUMO nodes until the next controlled light is reached

This contracts chains of unsignalized intersections:

$$
V = \{\text{controlled TLS ids}\}
(i,j) \in E
\iff
j \text{ is reachable from } i \text{ before hitting any other controlled TLS}
$$

The edge distance is the shortest walked distance, and the lane attribute is the bottleneck lane count on that walked path:

$$
d_{ij} = \min_{\rho:i \leadsto j} \sum_{(u,v)\in \rho} \mathrm{len}(u,v)
\ell_{ij} = \min_{(u,v)\in \rho^\star} \mathrm{lanes}(u,v)
$$

where $\rho^\star$ is the selected shortest path.

### `all_intersections`

- Nodes: every non-internal SUMO node
- Edges: immediate graph adjacency among all SUMO intersections
- RL agent IDs are attached back to the correct node IDs through `agent_node_indices` and `agent_node_mask`

So now:

$$
V = \{\text{all SUMO intersections}\}
$$
$$
E = \{\text{immediate node-to-node road connections}\}
$$

This is the only mode where graph nodes and RL agents are different objects.

## 3. Edge Features

Every graph edge may carry:

$$
e_{ij} = [d_{ij}, \ell_{ij}]
$$

where:

- $d_{ij}$ is road distance
- $\ell_{ij}$ is number of connecting lanes, or a bottleneck lane count in `walk_to_light`

These are passed into PyG `GATv2Conv(edge_dim=2)`.

## 4. From Raw Nodes To Agent Embeddings

Once the graph exists, the code runs a 2-layer GATv2 encoder.

### 4.1 Message passing view

At a high level, the encoder computes:

$$
h_i^{(0)} = W_{\text{in}} x_i + b_{\text{in}}
$$

$$
h_i^{(\ell+1)} =
\phi^{(\ell)}
\left(
h_i^{(\ell)},
\bigoplus_{j \in \mathcal N(i)}
\alpha_{ij}^{(\ell)}
\psi^{(\ell)}(h_j^{(\ell)}, e_{ij})
\right)
$$

where:

- $\mathcal N(i)$ is the graph neighborhood of node $i$
- $\alpha_{ij}^{(\ell)}$ is attention weight from `GATv2Conv`
- $\phi^{(\ell)}$ and $\psi^{(\ell)}$ are the learned GATv2 transformations
- $\bigoplus$ denotes attention-weighted aggregation

### 4.2 Exact repository structure

The code in `models/graph_encoder.py` is:

$$
h^{(0)} = \mathrm{ELU}(W_{\text{in}} x)
$$
$$
h^{(1)} = \mathrm{ELU}\big(\mathrm{GATv2Conv}_1(h^{(0)}, E, e)\big)
$$
$$
h^{(2)} = \mathrm{ELU}\big(\mathrm{GATv2Conv}_2(h^{(1)}, E, e)\big)
$$
$$
z = W_{\text{out}} h^{(2)}
$$

So each node gets a latent embedding $z_i \in \mathbb R^{d_z}$.

### 4.3 What happens in `all_intersections`

If graph nodes are not the same as RL agents, the code pools node embeddings back to each RL agent.

If agent $m$ is attached to node set $S_m \subseteq V$, then:

$$
z_m^{\text{agent}} = \frac{1}{|S_m|} \sum_{i \in S_m} z_i
$$

This is implemented in `MARLDiscreteSAC._pool_agent_latents(...)`.

That mean-pooling step is the bridge that makes `all_intersections` trainable without changing the action space.

## 5. Actor: Shared Per-Agent Policy

The actor is a shared MLP over each agent latent:

$$
\ell_i = f_\pi(z_i)
$$

where $\ell_i \in \mathbb R^{|\mathcal A|}$ is the logits vector over discrete green-phase actions.

The current code is:

$$
f_\pi(z_i) =
W_2 \,\mathrm{ReLU}(W_1 z_i + b_1) + b_2
$$

### 5.1 Action masking

The environment computes a legal action mask $m_i(a) \in \{0,1\}$ using:

- minimum green duration
- whether the signal is currently in a transitional phase
- green/yellow/all-red transition bookkeeping

The actor applies masking by replacing illegal logits with a very large negative number:

$$
\tilde \ell_i(a) =
\begin{cases}
\ell_i(a), & m_i(a)=1 \\
-10^8, & m_i(a)=0
\end{cases}
$$

Then:

$$
\pi_i(a \mid s) = \mathrm{softmax}(\tilde \ell_i)_a
$$

During training rollout the code samples from `Categorical(logits=...)`.
During deterministic evaluation it uses `argmax`.

## 6. Critic: Centralized Twin Q Networks

The critic is where CTDE is explicit.

First, compute a global graph context by mean-pooling agent latents:

$$
c = \frac{1}{N_a} \sum_{i=1}^{N_a} z_i
$$

Then each agent-specific critic input is:

$$
u_i = [z_i ; c]
$$

Each Q-network outputs Q-values for all discrete actions:

$$
Q_k(i, \cdot) = f_{Q_k}(u_i), \qquad k \in \{1,2\}
$$

The repository uses twin critics:

$$
Q_1, Q_2
$$

to reduce positive bias, and each is a 3-layer MLP.

This is centralized because the pooled context $c$ depends on the whole multi-agent state, not only agent $i$.

## 7. Why This Counts As CTDE Here

The implementation follows CTDE in this specific sense:

### Centralized during training

- the replay transition stores the whole graph state
- the encoder processes the whole graph
- the critic uses a pooled multi-agent context
- the target values depend on the policy distribution over all agents' local actions

### Decentralized at the action head

- the same actor network is applied independently to each agent latent
- action outputs remain per-agent
- each agent receives only its own masked categorical distribution

### Nuance

The actor latent $z_i$ is not built from $x_i$ alone.
It comes from graph message passing over the current graph observation.
So the execution policy is decentralized in output structure, but not strictly local-information-only.

That nuance is important if you compare this code to stricter Dec-POMDP formulations.

## 8. Discrete SAC Objective In This Repository

Let:

- $r_i$ be the per-agent reward from the environment
- $d_i$ be the done flag
- $\alpha = \exp(\log \alpha)$ be the entropy temperature

### 8.1 Target state value

For the next state:

$$
V(s') =\sum_a\pi(a \mid s')\Big(\min(Q_1'(s',a), Q_2'(s',a)) - \alpha \log \pi(a \mid s')\Big)
$$

In code this is computed independently for each agent and action dimension.

### 8.2 Critic target

$$
y_i = r_i + (1 - d_i)\gamma V_i(s')
$$

### 8.3 Critic loss

$$
\mathcal{L}_Q=\frac{1}{2}\left(\|Q_1(s,a) - y\|_2^2+\|Q_2(s,a) - y\|_2^2\right)
$$

This matches `DiscreteSACLossComputer._critic_loss(...)`.

### 8.4 Actor loss

The actor minimizes:

$$
\mathcal{L}_{\pi}=\mathbb{E}_{s}
\left[
\frac{1}{N_a}
\sum_{i=1}^{N_a}
\sum_{a}
\pi_i(a \mid s)
\Big(
\alpha \log \pi_i(a \mid s) - \min\left(Q_{1,i}(s,a), Q_{2,i}(s,a)\right)
\Big)
\right]$$

This is the exact expectation form used by Discrete SAC in the code.

### 8.5 Temperature loss

The implementation uses:

$$
\mathcal{L}_{\alpha}=- \log \alpha \, \big(\mathcal{H}(\pi) - \mathcal{H}_{\text{target}}\big)
$$

with target entropy heuristic:

$$
\mathcal{H}_{\text{target}} = -0.98\,|\mathcal{A}|
$$

## 9. Batched Graph Encoding Trick

PyG expects one graph at a time, but replay produces a batch of transitions.

So the implementation builds one block-diagonal super-graph:
$$
\tilde X =
\begin{bmatrix}
X^{(1)} \\
X^{(2)} \\
\vdots \\
X^{(B)}
\end{bmatrix}
$$

and shifts the edge indices by graph offset:
$$
\tilde E =
\{
(u + bN_g, v + bN_g)
\;|\;
(u,v)\in E,\;
b=0,\dots,B-1
\}
$$
where:

- $B$: replay batch size
- $N_g$: number of graph nodes in one environment state

This is exactly what `_batch_edge_index(...)` in `rl/losses.py` does.

## 10. Training Algorithm In Pseudocode

The current standalone training path is `scripts/train_gat_baseline.py`.

It trains episode-by-episode, not step-by-step online.

```text
Algorithm 1: Current training loop in scripts/train_gat_baseline.py

Input:
  env config, model config, optimizer config
  replay capacity C
  warmup threshold W
  updates per transition U
  batch size B

Initialize SUMO environment
Reset env once to infer:
  obs_dim, num_actions, n_agents, graph topology

Initialize:
  graph encoder f_G
  shared actor pi_theta
  twin critics Q_phi1, Q_phi2
  target critic Q'_phi
  temperature parameter log_alpha
  replay buffer D

for episode = 1 ... E do
  Set agent to training mode
  Reset env
  transitions = []

  while episode not done do
    Build current tensors:
      agents/observation
      graph_observation
      agent_node_indices, agent_node_mask
      edge_index, edge_attr
      action_mask

    Encode graph:
      z_nodes = f_G(graph_observation, edge_index, edge_attr)
      z_agents = pool(z_nodes, agent_node_indices, agent_node_mask)

    Sample masked actions from shared actor
    Step SUMO for delta_t seconds
    Receive reward, done, next observation
    Pack transition and append to episode list
  end while

  Push all episode transitions to replay D

  if |D| >= W then
    Repeat U * (#new transitions) times:
      sample batch from D
      build block-diagonal graph batch
      compute critic loss
      update critic
      compute actor loss
      update encoder + actor
      compute alpha loss
      update log_alpha
    end repeat

    soft-update target critic once
  end if

  if episode return is best so far:
    save best_agent.pt
end for
```

### Important implementation details

- replay is uniform random sampling
- the actor optimizer also updates the encoder
- the critic optimizer updates only the critic
- actor and critic gradients are clipped to norm 10
- target critic is soft-updated once per episode after the episode's update block

## 11. Running / Deployment Algorithm In Pseudocode

The same forward path is used in rollout and evaluation, but without gradient updates.

```text
Algorithm 2: Current execution loop

Reset env

while not done do
  Build:
    graph_observation
    edge_index, edge_attr
    agent_node_indices, agent_node_mask
    action_mask

  z_nodes   = encoder(graph_observation, edge_index, edge_attr)
  z_agents  = pool(z_nodes, agent_node_indices, agent_node_mask)
  logits    = shared_actor(z_agents)
  logits    = apply_action_mask(logits, action_mask)

  if deterministic:
    a_i = argmax logits_i
  else:
    a_i ~ Categorical(logits_i)

  Env applies legal phase transitions:
    - respects min_green_duration
    - uses inferred yellow phases
    - uses inferred all-red phases

  Advance SUMO by delta_t seconds
  Read next state
end while
```

## 12. What Changes Across Graph Modes

### `original`

- smallest graph
- node count equals number of RL agents
- no agent-node pooling needed in practice

### `walk_to_light`

- still node count equals number of RL agents
- graph is denser because unsignalized chains are contracted
- checkpoint-compatible with `original` as long as observation dimension stays the same

### `all_intersections`

- graph node count is larger than RL agent count
- graph observation and agent observation are different tensors
- agent embeddings are pooled from node embeddings
- old checkpoints from the smaller node-only-TLS modes will often be incompatible because input dimension and graph semantics change

## 13. The Best Mental Model

The most faithful way to think about the current system is:

1. Build a traffic graph from SUMO.
2. Put traffic measurements on graph nodes.
3. Run graph attention to compute one latent per node.
4. Convert node latents into one latent per RL-controlled traffic light.
5. Use that latent twice:
   - once locally for the shared actor
   - once jointly through mean pooling for the centralized twin critic
6. Train with Discrete SAC from replay.

That is the whole architecture.

The graph encoder is the mechanism that says "my state should already contain neighborhood structure."
The centralized critic is the mechanism that says "training should still reason about multi-agent coupling."

