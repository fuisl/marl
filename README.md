# Graph-Based Multi-Agent Reinforcement Learning for Traffic Signal Control: A Library Analysis and Implementation Plan
* **SUMO / TraCI**: simulation and online traffic-light control.
* **TorchRL**: MARL data structure, replay, losses, and algorithm plumbing.
* **PyG**: graph representation learning over the road network.
* **Lightning**: experiment shell and manual multi-optimizer training loop.
* **Optional sumo-rl / PettingZoo**: reference environment structure, not your final core abstraction. ([Eclipse SUMO][1])

Below is the grounded analysis first, then the final implementation plan.

---

## 1. What each library is actually good for here

### SUMO / TraCI

SUMO’s TraCI API is designed for online control of a running traffic simulation. It lets you query simulation state and change traffic-light behavior during execution; the official docs specifically expose operations such as `setPhase`, `setProgram`, and `setPhaseDuration`. SUMO also notes that if performance becomes a bottleneck, **libsumo** is a drop-in faster alternative because it shares the same function signatures. ([Eclipse SUMO][1])

**Reliable pattern for your project:**
Use SUMO as the source of truth for traffic dynamics, and keep all RL logic outside it. Your environment wrapper should:

1. step the simulator,
2. read detector/lane/intersection features,
3. translate them into tensors,
4. apply actions back through TraCI.

That is the cleanest boundary.

### TorchRL

TorchRL’s multi-agent docs are actually a very good fit for traffic control. The official MARL API is built around a nested `TensorDict`, where agent-specific values live under an `"agents"` entry and shared values remain at the top level. TorchRL also supports grouping agents so they can be processed together, which is exactly what you want for homogeneous traffic-light controllers with shared parameters. The docs and examples also show explicit support for **parameter sharing** and **centralized critics**, which are the right patterns for CTDE traffic MARL. ([PyTorch Documentation][2])

TorchRL also gives you:

* `TensorDictReplayBuffer` for off-policy storage,
* actor and probabilistic actor wrappers,
* SAC and Discrete SAC loss modules. ([PyTorch Documentation][3])

**Reliable pattern for your project:**
Use TorchRL for:

* MARL-shaped tensors,
* replay buffer,
* actor/critic wrappers,
* SAC losses,
* target updates and training mechanics.

Do **not** try to force TorchRL to be your graph library.

### PyTorch Geometric

PyG is the correct place for the road-network encoder. The docs support both:

* homogeneous graph data,
* heterogeneous graph data via `HeteroData`,
* heterogeneous message passing via `HeteroConv`,
* typed transformer attention via `HGTConv`. ([PyTorch Geometric][4])

For attention specifically:

* `GATConv` supports edge features via `edge_dim`,
* `GATv2Conv` improves over standard GAT by fixing the “static attention” issue,
* `HGTConv` is the principled choice when you move to typed nodes/edges. ([PyTorch Geometric][5])

**Reliable pattern for your project:**
Use PyG **only** for the encoder:

* static road topology,
* dynamic node features,
* optional edge features,
* output one latent embedding per controlled intersection.

Do **not** put replay or RL losses into PyG abstractions.

### Lightning

Lightning’s own docs are explicit that **manual optimization** is the right mode for advanced research cases like RL, especially when multiple optimizers are involved. In manual mode, Lightning still handles accelerators, precision, and strategy, while you control optimizer stepping and backward passes yourself. ([Lightning AI][6])

**Reliable pattern for your project:**
Use Lightning as the training shell:

* checkpointing,
* logging,
* evaluation hooks,
* multi-optimizer stepping,
* experiment reproducibility.

Do **not** try to express the SAC update as standard supervised `training_step` logic with automatic optimization.

### sumo-rl / PettingZoo

The `sumo-rl` project is useful because it already exposes a SUMO traffic signal control environment compatible with Gymnasium and PettingZoo, and its environment code shows the standard responsibilities of a SUMO traffic RL wrapper. TorchRL also has a `PettingZooEnv` wrapper for PettingZoo environments. ([GitHub][7])

**Reliable pattern for your project:**
Treat `sumo-rl` as a **reference implementation** for:

* environment lifecycle,
* agent IDs,
* signal timing mechanics,
* observation/reward plumbing.

I would not base your final research system entirely on `sumo-rl` unless your method is very close to its assumptions.

---

## 2. What the docs imply for your MARL traffic architecture

The official examples point to a consistent design:

### Pattern A: synchronous multi-agent stepping

TorchRL’s MARL tutorials assume a setting where all agents act synchronously at each step. Traffic signal control matches that naturally if you make each control decision every fixed interval, such as every 5–10 simulated seconds. ([GitHub][8])

### Pattern B: shared-parameter agents

`MultiAgentMLP` in TorchRL explicitly supports `share_params=True`, which is the standard pattern for homogeneous agents like intersections of the same controller type. ([PyTorch Documentation][9])

For your case, even if you do not end up using `MultiAgentMLP` itself, the **pattern** is correct:

* one actor shared across intersections,
* maybe one shared pair of critics,
* node embeddings differ per agent because the graph encoder sees different local states.

### Pattern C: centralized critic

TorchRL’s MARL tutorials explicitly highlight centralized critic architectures. That is exactly what you want under CTDE:

* actor uses local or graph-local information,
* critic uses richer joint or pooled information during training. ([PyTorch Documentation][10])

### Pattern D: discrete or continuous SAC

TorchRL supports both SAC and Discrete SAC. Since traffic signals are usually easier to control with discrete choices such as keep/switch/select-phase, **Discrete SAC** is the most natural first target. ([PyTorch Documentation][11])

---

## 3. Grounded verdict on algorithm choices

## TorchRL for MARL

**Yes, use it.**
The official tutorials and API design support exactly the three things you need:

* multi-agent nested data,
* shared parameter policies,
* centralized critics. ([PyTorch Documentation][2])

### Best pattern

Use TorchRL with a custom environment that outputs:

* shared graph tensors at top level,
* agent tensors under `"agents"`.

Then use TorchRL for:

* replay buffer,
* action sampling interface,
* SAC loss modules.

### What not to do

Do not rely too heavily on TorchRL’s stock network modules for the final actor backbone, because your real backbone is graph-based and belongs in PyG. TorchRL should wrap your graph-backed actor, not replace it.

---

## 4. Grounded verdict on PyG for graph traffic state

**Yes, use it.**
PyG is the correct place to encode network structure.

### Start simple

Use a **homogeneous intersection graph** first:

* one node per controlled traffic light,
* edges for adjacency / upstream-downstream coordination,
* node features from SUMO,
* optional edge features such as distance, road capacity, or lane count.

This aligns with `GATConv`/`GATv2Conv` patterns and is much easier to debug. PyG’s GAT variants support edge features if needed. ([PyTorch Geometric][5])

### When to move to heterogeneity

Use `HeteroData` + `HGTConv` only when your problem genuinely needs multiple entity types, for example:

* intersection nodes,
* lane nodes,
* phase nodes,
* typed relations like `incoming_to`, `controls`, `conflicts_with`. ([PyTorch Geometric][4])

### My judgment

For a first publishable system, **GATv2 on an intersection graph** is the best starting point. The docs themselves note GATv2 addresses a limitation of standard GAT’s static attention, so it is the better default if you want node-conditioned neighbor ranking. ([PyTorch Geometric][12])

---

## 5. Grounded verdict on Lightning

**Yes, but narrowly.**
Use Lightning for the training shell only. The official docs are very clear that manual optimization is the intended route for RL with multiple optimizers. ([Lightning AI][6])

### Best pattern

Your Lightning module should:

* own actor, critics, target critics, alpha parameter,
* sample from replay,
* compute losses,
* call manual backward,
* step optimizers in the right order,
* log metrics.

### What not to do

Do not let Lightning own the rollout loop in a rigid dataloader-like way. Rollout collection from SUMO is custom and simulator-bound.

---

## 6. Grounded verdict on SUMO integration

There are two reliable integration patterns.

### Pattern 1: Custom wrapper over TraCI

This gives full control over:

* timing logic,
* reward design,
* graph construction,
* signal constraints.

This is the best final architecture.

### Pattern 2: PettingZoo-compatible environment

The `sumo-rl` project shows this pattern, and TorchRL can wrap PettingZoo environments directly. This is the fastest way to get something running. ([GitHub][7])

### My judgment

Use **Pattern 1** for the final implementation, but borrow structure from Pattern 2.

---

## 7. The reliable architecture pattern for your project

Here is the architecture I recommend after looking at the docs and examples.

### Environment state representation

At decision time (t), build:

* node feature matrix `x_t` of shape `[n_agents, d_node]`,
* fixed `edge_index`,
* optional `edge_attr`.

Node features should include:

* queue lengths by approach,
* waiting times,
* occupancy / density,
* current phase one-hot,
* elapsed green time,
* pressure,
* arrivals or departures over the last window.

These are not library constraints; they are the practical traffic features the architecture needs.

### Graph encoder

Use PyG:

* `GATv2Conv` for V1,
* `HGTConv` only for later hetero upgrade.

Output:

* per-agent latent `z_i`.

### Actor

Shared actor over all intersections:

* input: `z_i`,
* output: discrete phase action logits.

Use TorchRL probabilistic actor wrappers around this module if you want native TorchRL action handling. TorchRL’s actor module stack is built for wrapping custom `TensorDictModule`-style modules. ([PyTorch Documentation][13])

### Critic

Twin critics for SAC:

* input: `z_i`, action, and pooled context `c`,
* or neighbor-aware context.

The centralized-critic pattern is grounded in TorchRL’s MARL examples and `MultiAgentMLP` design, where centralized models can consume all-agent inputs. ([PyTorch Documentation][10])

### Replay

Use `TensorDictReplayBuffer`.
Store:

* current node features,
* next node features,
* actions,
* rewards,
* done flags,
* maybe phase mask / legal action mask.

Keep topology fixed outside replay whenever possible. TorchRL replay is good at tensordicts; it does not buy you anything to serialize complicated Python graph objects if your road network is static. ([PyTorch Documentation][3])

---

## 8. What the examples suggest you should not overcomplicate

A common trap is trying to make everything “native” to one framework. The docs suggest a better split.

### Do not make TorchRL own the graph object

TorchRL wants tensors in tensordicts. PyG wants graph data structures. The reliable pattern is:

* store tensors in replay,
* reconstruct graph inputs at forward time,
* keep static topology cached in the model or env.

### Do not start with heterogeneous graph MARL SAC all at once

PyG supports heterogeneity, but the docs also make clear that hetero graphs are a materially more complex representation. Start with homogeneous node features unless your novelty depends on typed relations from day one. ([PyTorch Geometric][4])

### Do not start with continuous duration control

SAC supports continuous control, but for traffic signals, discrete action spaces are easier to stabilize and match the real signal logic better. Since TorchRL already has `DiscreteSACLoss`, that is the cleaner first implementation. ([PyTorch Documentation][11])

---

## 9. Final implementation plan

This is the plan I would actually execute.

## Phase 1 — Environment and state pipeline

### 1. Build a custom SUMO environment

Implement `TrafficSignalEnv` around TraCI/libsumo.

Responsibilities:

* launch SUMO,
* maintain agent list = traffic light IDs,
* advance simulation for `delta_t` seconds per decision,
* expose observations and rewards,
* apply actions with `setPhase` or `setPhaseDuration`,
* enforce yellow/all-red transition rules internally. SUMO explicitly supports changing phase index and phase duration online. ([Eclipse SUMO][1])

### 2. Define the MARL output format

Output a TorchRL-style tensordict:

* top level:

  * `edge_index`
  * `edge_attr` if used
  * shared metrics
* under `"agents"`:

  * `observation`
  * `reward`
  * `done`
  * `action_mask` if needed

This matches TorchRL’s official MARL structure with shared keys at the top level and agent-specific keys in the nested agent tensordict. ([PyTorch Documentation][2])

### 3. Build a graph map once

From the SUMO road network:

* node = each controlled intersection,
* edge = neighboring / traffic-coupled intersections,
* save `edge_index` once.

This graph should stay fixed during training unless the network itself changes.

---

## Phase 2 — First working model

### 4. Use a homogeneous intersection graph

Do **not** start with `HeteroData`.

Use:

* `x`: `[n_agents, d_node]`
* `edge_index`
* optional `edge_attr`

### 5. Implement the graph encoder in PyG

Start with:

* `Linear(d_node, hidden)`
* `GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim)`
* `GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim)`
* output projection to latent `z_i`

Why `GATv2Conv` first:

* attention over neighbors,
* better default than standard GAT,
* still simple,
* supports edge features. ([PyTorch Geometric][12])

### 6. Shared discrete actor

Actor head:

* input: `z_i`
* output: logits over legal phase actions

Use shared parameters across all intersections. That matches TorchRL’s multi-agent shared-parameter pattern. ([PyTorch Documentation][9])

### 7. Centralized twin critics

Critic input:

* local latent `z_i`,
* chosen action `a_i`,
* pooled graph context `c = mean(z)` or attention readout.

This is the simplest graph-aware CTDE critic.

---

## Phase 3 — RL plumbing with TorchRL

### 8. Wrap the custom modules in TorchRL-compatible modules

Use TensorDict-compatible wrappers so your actor and critics read/write from tensordicts. TorchRL’s actor stack is designed around `TensorDictModule` and probabilistic actors. ([PyTorch Documentation][14])

### 9. Use `DiscreteSACLoss`

Since actions are discrete phase choices, use TorchRL’s `DiscreteSACLoss`. The docs explicitly support discrete action spaces and adaptive alpha loss. ([PyTorch Documentation][11])

### 10. Use `TensorDictReplayBuffer`

Replay stores:

* `agents.observation`
* `agents.action`
* `agents.reward`
* `agents.done`
* `next.agents.observation`
* any masks needed
* optionally shared graph fields if not fixed globally

This is exactly what TorchRL replay is built for. ([PyTorch Documentation][3])

### 11. Group all traffic lights into one homogeneous agent group

If you use TorchRL grouping, put all traffic lights into the same group so the policy is vectorized and parameter-shared. This aligns with TorchRL’s `group_map` design. ([PyTorch Documentation][15])

---

## Phase 4 — Training shell with Lightning

### 12. Use manual optimization

In Lightning:

* `self.automatic_optimization = False`
* separate optimizers for actor, critics, alpha
* call `zero_grad`, `manual_backward`, `step` yourself

This follows the official Lightning RL-relevant pattern. ([Lightning AI][6])

### 13. Keep rollout collection outside supervised dataloaders

Use a custom rollout loop:

* interact with SUMO,
* append to replay,
* sample minibatches,
* optimize.

Lightning should orchestrate the experiment, not define the simulator loop.

### 14. Log real traffic metrics

Log at least:

* episodic return,
* average waiting time,
* average queue length,
* average travel time,
* throughput,
* number of stops.

Reward alone is not enough in traffic control.

---

## Phase 5 — Upgrade path after the baseline works

### 15. Upgrade to hetero graph only if baseline saturates

Move to `HeteroData` + `HGTConv` when you need:

* lane nodes,
* phase nodes,
* conflict edges,
* typed road relations.

PyG’s hetero docs and `HGTConv` support this cleanly. ([PyTorch Geometric][4])

### 16. Consider libsumo for throughput

If simulator overhead dominates, switch from TraCI to libsumo. SUMO explicitly says the signatures are the same, so this is a practical performance upgrade path. ([Eclipse SUMO][1])

### 17. Only then explore continuous SAC or hybrid actions

After the discrete system is stable, experiment with:

* duration adjustment,
* split optimization,
* hybrid phase + duration.

---

## 10. Recommended repository structure

```text
project/
  env/
    sumo_env.py
    traci_adapter.py
    graph_builder.py
    reward.py
    action_constraints.py
  models/
    graph_encoder.py
    actor.py
    critic.py
    marl_discrete_sac.py
  rl/
    replay.py
    rollout.py
    losses.py
  train/
    lightning_module.py
    train.py
    evaluate.py
  configs/
    env.yaml
    model.yaml
    train.yaml
  scripts/
    build_network.py
    run_sumo_baseline.py
```

---

## 11. Final recommended V1 stack

This is the version I recommend you implement first:

* **Environment**: custom SUMO wrapper over TraCI
* **Graph**: homogeneous intersection graph
* **Encoder**: 2-layer `GATv2Conv`
* **Actor**: shared discrete policy over phase actions
* **Critic**: centralized twin critics with pooled graph context
* **RL**: TorchRL `DiscreteSACLoss` + `TensorDictReplayBuffer`
* **Training shell**: Lightning manual optimization
* **Reference only**: borrow structure from `sumo-rl` and TorchRL MARL tutorials

This plan is the most grounded in the existing library patterns, and it minimizes integration friction:

* SUMO owns simulation,
* PyG owns graph encoding,
* TorchRL owns MARL and SAC machinery,
* Lightning owns training orchestration. ([Eclipse SUMO][1])

The next useful step is to turn this plan into a **module-by-module code skeleton** with the actual class interfaces for `sumo_env.py`, `graph_encoder.py`, and the TorchRL-compatible actor/critic wrappers.

[1]: https://sumo.dlr.de/docs/TraCI/index.html "TraCI - SUMO Documentation"
[2]: https://docs.pytorch.org/rl/main/reference/envs_multiagent.html "Multi-agent Environments — torchrl main documentation"
[3]: https://docs.pytorch.org/rl/main/reference/generated/torchrl.data.TensorDictReplayBuffer.html "TensorDictReplayBuffer — torchrl main documentation"
[4]: https://pytorch-geometric.readthedocs.io/en/2.6.0/notes/heterogeneous.html "Heterogeneous Graph Learning — pytorch_geometric  documentation"
[5]: https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.GATConv.html "torch_geometric.nn.conv.GATConv — pytorch_geometric  documentation"
[6]: https://lightning.ai/docs/pytorch/stable//model/manual_optimization.html "Manual Optimization — PyTorch Lightning 2.6.1 documentation"
[7]: https://github.com/LucasAlegre/sumo-rl?utm_source=chatgpt.com "LucasAlegre/sumo-rl"
[8]: https://github.com/pytorch/rl/blob/main/tutorials/sphinx-tutorials/multiagent_competitive_ddpg.py "rl/tutorials/sphinx-tutorials/multiagent_competitive_ddpg.py at main · pytorch/rl · GitHub"
[9]: https://docs.pytorch.org/rl/0.7/reference/generated/torchrl.modules.MultiAgentMLP.html "MultiAgentMLP — torchrl 0.7 documentation"
[10]: https://docs.pytorch.org/rl/main/tutorials/multiagent_ppo.html "Multi-Agent Reinforcement Learning (PPO) with TorchRL Tutorial — torchrl main documentation"
[11]: https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.DiscreteSACLoss.html?utm_source=chatgpt.com "DiscreteSACLoss — torchrl main documentation"
[12]: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.GATv2Conv.html "torch_geometric.nn.conv.GATv2Conv — pytorch_geometric  documentation"
[13]: https://docs.pytorch.org/rl/stable/reference/modules_actors.html "Actor Modules — torchrl 0.11 documentation"
[14]: https://docs.pytorch.org/rl/stable/reference/modules.html?utm_source=chatgpt.com "torchrl.modules package"
[15]: https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.MarlGroupMapType.html?utm_source=chatgpt.com "MarlGroupMapType — torchrl main documentation"
