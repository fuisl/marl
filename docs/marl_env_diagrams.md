# Current MARL Implementation Diagrams

See also: [CTDE Graph-SAC Architecture](./ctde_graph_sac_architecture.md)

This report documents the current end-to-end architecture and the baseline training path used in this repository.

## 1. System Context in the Project

```mermaid
flowchart TB
  Hydra[Hydra Configs\nconfigs/*.yaml]

  subgraph EntryPoints[Entry Points]
    GATBaseline[scripts/train_gat_baseline.py]
    LightningTrain[train/train.py]
    Eval[train/evaluate.py]
    SUMOBaseline[scripts/run_sumo_baseline.py]
  end

  subgraph TrainStack[Training Stack]
    Lightning[TrafficMARLModule\ntrain/lightning_module.py]
    ManualLoop[Manual SAC Loop\nReplay + Update\nscripts/train_gat_baseline.py]
    Rollout[RolloutWorker\nrl/rollout.py]
    Replay[TensorDictReplayBuffer\nrl/replay.py]
    Loss[DiscreteSACLossComputer\nrl/losses.py]
    Opt[make_optimizer\nAdam or MetaAdam\nrl/optimizers.py]
  end

  subgraph ModelStack[Model Stack]
    Agent[MARLDiscreteSAC]
    Encoder[GraphEncoder\nGATv2]
    Actor[SharedDiscreteActor]
    Critic[CentralizedTwinCritic]
  end

  subgraph EnvPkg[marl_env package]
    Env[TrafficSignalEnv]
    AC[ActionConstraints]
    GB[GraphBuilder]
    RC[RewardCalculator]
    TA[TraCIAdapter]
  end

  SUMO[SUMO Simulator\nTraCI or libsumo]

  Hydra --> GATBaseline
  Hydra --> LightningTrain
  Hydra --> Eval
  Hydra --> SUMOBaseline

  GATBaseline --> ManualLoop
  LightningTrain --> Lightning
  Eval --> Lightning
  SUMOBaseline --> Env

  Lightning --> Rollout
  Lightning --> Replay
  Lightning --> Loss
  Lightning --> Agent

  ManualLoop --> Replay
  ManualLoop --> Loss
  ManualLoop --> Agent
  ManualLoop --> Opt

  Rollout --> Env
  Rollout --> Agent
  Replay --> Loss
  Loss --> Agent

  Agent --> Encoder
  Agent --> Actor
  Agent --> Critic

  Env --> AC
  Env --> GB
  Env --> RC
  Env --> TA
  TA <--> SUMO
```

## 2. Baseline Architecture (Current)

The current baseline is `scripts/train_gat_baseline.py` with Hydra-driven config, graph encoder + Discrete SAC, and optimizer selection through `rl/optimizers.py`.

```mermaid
flowchart LR
  Cfg[configs/gat_baseline.yaml\n+ env/default\n+ model/default] --> Train[scripts/train_gat_baseline.py]
  Train --> Env[TrafficSignalEnv]
  Train --> Agent[MARLDiscreteSAC]
  Train --> Loss[DiscreteSACLossComputer]
  Train --> RB[ReplayBuffer\nFIFO ring buffer]
  Train --> Opt[make_optimizer]

  Agent --> Enc[GraphEncoder\n2x GATv2Conv]
  Agent --> Act[SharedDiscreteActor]
  Agent --> Crit[CentralizedTwinCritic\n+ target critic]

  Opt --> Adam[Adam]
  Opt --> Meta[MetaAdam\nscalar hypergradient LR]

  Env --> SUMO[SUMO/libsumo via TraCIAdapter]
  Loss --> Agent
  RB --> Loss
```

## 3. Baseline Training Loop (One Episode + Updates)

```mermaid
sequenceDiagram
  participant T as train_gat_baseline.py
  participant E as TrafficSignalEnv
  participant A as MARLDiscreteSAC
  participant R as ReplayBuffer
  participant L as DiscreteSACLossComputer
  participant O as Optimizers (Adam/MetaAdam)

  T->>E: reset()
  loop until done
    T->>A: select_action(obs, edge_index, edge_attr, action_mask)
    A-->>T: actions
    T->>E: step(actions)
    E-->>T: next_td (reward, done, next obs)
    T->>R: push(pack_transition(...))
    alt replay has enough samples
      T->>R: sample(batch)
      R-->>T: transitions
      T->>L: compute(batch)
      L-->>T: critic_loss, actor_loss, alpha_loss
      T->>O: critic step
      T->>O: actor step
      T->>O: alpha step
      T->>A: soft_update_target()
    end
  end
```

## 4. `marl_env` Internal Module Dependency Map

```mermaid
flowchart LR
  subgraph marl_env
    sumo_env[sumo_env.py\nTrafficSignalEnv]
    action_constraints[action_constraints.py\nActionConstraints]
    traci_adapter[traci_adapter.py\nTraCIAdapter]
    graph_builder[graph_builder.py\nGraphBuilder]
    reward[reward.py\nRewardCalculator]
  end

  sumo_env --> action_constraints
  sumo_env --> traci_adapter
  sumo_env --> graph_builder
  sumo_env --> reward

  traci_adapter --> traci[traci or libsumo]
  graph_builder --> sumolib[sumolib]
```

## 5. Core Class Diagram

```mermaid
classDiagram
  class TrafficSignalEnv {
    +reset() TensorDict
    +step(actions: Tensor) TensorDict
    +close() None
    +observation_dim int
    +num_actions int
    -_build_tensordict() TensorDict
    -_get_observation(tl_id: str) Tensor
    -_compute_rewards() Tensor
    -_apply_transitions() None
    -_extract_green_phases(tl_id: str) list[int]
    -_build_transition_maps(tl_id: str, num_phases: int)
  }

  class TraCIAdapter {
    +start() None
    +close(wait=False) None
    +simulation_step() None
    +get_traffic_light_ids() list[str]
    +get_phase(tl_id: str) int
    +set_phase(tl_id: str, phase_index: int) None
    +get_program_logic(tl_id: str) Any
    +get_controlled_lanes(tl_id: str) list[str]
    +current_time float
    +min_expected_vehicles int
  }

  class GraphBuilder {
    +build() edge_index_and_optional_edge_attr
    +edge_index Tensor
    +edge_attr optional Tensor
    +num_nodes int
    -_get_neighbor_tl_ids(node) list[(id,dist,lanes)]
  }

  class RewardCalculator {
    +mode str
    +weights dict[str,float]
    +compute(metrics: IntersectionMetrics) float
    +compute_batch(metrics_list: list[IntersectionMetrics]) Tensor
  }

  class IntersectionMetrics {
    +queue_lengths list[float]
    +waiting_times list[float]
    +mean_speeds list[float]
    +occupancies list[float]
    +throughput int
  }

  class ActionConstraints {
    +register_agent(...)
    +get_action_mask(tl_id, current_green_phase, elapsed_green) Tensor
    +begin_switch(tl_id, current_green_phase, target_action) None
    +phase_to_apply(tl_id) optional int
    +tick(tl_id, seconds=1) bool
    +destination_green(tl_id) optional int
    +complete_switch(tl_id) None
    +in_transition(tl_id) bool
    +action_to_green_phase(tl_id, action_idx) int
    +green_phase_to_action(tl_id, green_phase) int
  }

  class TransitionPlan {
    +from_green_phase int
    +to_green_phase int
    +yellow_phase optional int
    +all_red_phase optional int
    +stage str
    +timer int
  }

  class _AgentState {
    +num_phases int
    +green_phase_indices list[int]
    +action_to_green dict[int,int]
    +green_to_action dict[int,int]
    +yellow_phase_map dict[(int,int),int]
    +all_red_phase_map dict[(int,int),int]
    +transition optional TransitionPlan
  }

  TrafficSignalEnv --> TraCIAdapter
  TrafficSignalEnv --> GraphBuilder
  TrafficSignalEnv --> RewardCalculator
  TrafficSignalEnv --> ActionConstraints
  RewardCalculator --> IntersectionMetrics
  ActionConstraints --> _AgentState
  _AgentState --> TransitionPlan
```

## 6. `reset()` Runtime Sequence

```mermaid
sequenceDiagram
  participant C as Caller
  participant E as TrafficSignalEnv
  participant T as TraCIAdapter
  participant G as GraphBuilder
  participant A as ActionConstraints

  C->>E: reset()
  E->>T: close() (no-op if first call)
  E->>T: start()
  E->>T: get_traffic_light_ids()
  T-->>E: tl_ids

  E->>G: GraphBuilder(net_file, tl_ids)
  E->>G: build()
  G-->>E: edge_index, edge_attr

  loop each tl_id
    E->>T: get_program_logic(tl_id)
    E->>E: _extract_green_phases()
    E->>E: _build_transition_maps()
    E->>T: get_controlled_lanes(tl_id)
    E->>T: get_phase(tl_id)
    E->>A: register_agent(...)
  end

  E->>E: compute max_lanes / max_green
  E->>E: _build_tensordict()
  E-->>C: TensorDict{agents/observation, agents/action_mask, edge_index, edge_attr?}
```

## 7. `step(actions)` Runtime Sequence

```mermaid
sequenceDiagram
  participant C as Caller
  participant E as TrafficSignalEnv
  participant A as ActionConstraints
  participant T as TraCIAdapter
  participant R as RewardCalculator

  C->>E: step(actions[n_agents])

  loop each tl_id
    E->>A: in_transition(tl_id)?
    alt not in transition
      E->>A: begin_switch(tl_id, current_green, action_idx)
    end
  end

  loop second in [1..delta_t]
    E->>E: _apply_transitions()
    loop each tl_id
      E->>A: in_transition(tl_id)?
      alt in transition
        E->>A: phase_to_apply(tl_id)
        A-->>E: phase | None
        alt phase is not None
          E->>T: set_phase(tl_id, phase)
        end

        E->>A: tick(tl_id, 1)
        alt transition done
          E->>A: destination_green(tl_id)
          A-->>E: dest_green
          E->>T: set_phase(tl_id, dest_green)
          E->>A: complete_switch(tl_id)
          E->>E: current_green=dest, elapsed_green=0
        end
      else no transition
        E->>E: elapsed_green += 1
      end
    end

    E->>T: simulation_step()
  end

  E->>E: _build_tensordict()
  E->>E: _compute_rewards()
  E->>R: compute_batch(metrics_list)
  R-->>E: rewards[n_agents]

  E->>T: current_time, min_expected_vehicles
  E-->>C: TensorDict + reward + done flags
```

## 8. Action Transition FSM (`ActionConstraints`)

```mermaid
stateDiagram-v2
  [*] --> NoTransition

  state NoTransition {
    [*] --> Ready
  }

  NoTransition --> NoTransition: begin_switch(same green)
  NoTransition --> Yellow: begin_switch(new green, yellow exists)
  NoTransition --> AllRed: begin_switch(new green, no yellow, all-red exists)
  NoTransition --> ReadyToCommit: begin_switch(new green, no clearance phases)

  Yellow --> Yellow: tick(); timer > 0
  Yellow --> AllRed: timer <= 0 and all_red_phase exists
  Yellow --> ReadyToCommit: timer <= 0 and no all_red_phase

  AllRed --> AllRed: tick(); timer > 0
  AllRed --> ReadyToCommit: timer <= 0

  ReadyToCommit --> NoTransition: env sets destination green + complete_switch()
```

## 9. Action-Mask Decision Logic

```mermaid
flowchart TD
  Start[Build mask for one agent] --> Q1{In transition?}

  Q1 -->|Yes| DestOnly[Mask all False except destination action]
  Q1 -->|No| Q2{elapsed_green < min_green_duration?}

  Q2 -->|Yes| KeepOnly[Mask all False except current green action]
  Q2 -->|No| AllOpen[All actions allowed]

  DestOnly --> End[Return bool mask]
  KeepOnly --> End
  AllOpen --> End
```

## 10. Observation Vector Construction

```mermaid
flowchart LR
  Lanes[Controlled lanes for tl_id] --> Q[queue per lane\nget_lane_halting_number]
  Lanes --> W[waiting time per lane\nget_lane_waiting_time]
  Lanes --> O[occupancy per lane\nget_lane_occupancy]
  Lanes --> S[mean speed per lane\nget_lane_mean_speed]

  Q --> Qp[pad/truncate to max_lanes]
  W --> Wp[pad/truncate to max_lanes]
  O --> Op[pad/truncate to max_lanes]
  S --> Sp[pad/truncate to max_lanes]

  Phase[current_green for tl_id] --> OneHot[phase one-hot\nsize=max_green]
  Elapsed[elapsed_green for tl_id] --> El[scalar tensor size=1]

  Qp --> Cat[concat]
  Wp --> Cat
  Op --> Cat
  Sp --> Cat
  OneHot --> Cat
  El --> Cat

  Cat --> Obs[observation_i\nshape: 4*max_lanes + max_green + 1]
```

## 11. Environment Output `TensorDict` Schema

```mermaid
flowchart TB
  TD[TensorDict\nroot batch]

  TD --> Agents[agents\nTensorDict for all agents]
  TD --> EdgeIndex[edge_index\nshape 2 by E]
  TD --> EdgeAttr[edge_attr\nshape E by d_edge\noptional]
  TD --> DoneRoot[done\nshape 1\nbool after step]

  Agents --> Obs[observation\nshape n_agents by obs_dim]
  Agents --> Mask[action_mask\nshape n_agents by num_actions]
  Agents --> Reward[reward\nshape n_agents by 1\nafter step]
  Agents --> DoneAgent[done\nshape n_agents by 1\nafter step]
```

## 12. Graph Topology Build Flow (`GraphBuilder`)

```mermaid
flowchart TD
  A[Inputs net_file and tl_ids] --> B[Load SUMO net]
  B --> C[Map tl_id -> node and index]

  C --> D{for each tl node}
  D --> E[scan outgoing edges]
  D --> F[scan incoming edges]

  E --> G[collect neighbor tl_id, edge length, lane count]
  F --> G
  G --> H[filter: neighbor must be in tl_ids and not seen]
  H --> I[append src_idx dst_idx and attrs]

  I --> J{any edges collected?}
  J -->|No| K[edge_index empty and no edge_attr]
  J -->|Yes| L[edge_index tensor 2 by E\nedge_attr tensor E by 2]

  K --> M[return edge_index, edge_attr]
  L --> M
```

## 13. Reward Computation Flow

```mermaid
flowchart TD
  A[Compute rewards in env] --> B{for each tl_id}
  B --> C[Read lane metrics from TraCIAdapter\nqueue, waiting, speed, occupancy]
  C --> D[Build IntersectionMetrics]
  D --> E[Append to metrics_list]
  E --> F[Batch reward aggregation]

  F --> G{mode}
  G -->|queue| H[queue penalty reward]
  G -->|wait| I[waiting time penalty reward]
  G -->|pressure| J[pressure proxy reward]
  G -->|combined| K[weighted combined reward]

  H --> Z[Tensor of n_agents]
  I --> Z
  J --> Z
  K --> Z
```

## 14. Environment Lifecycle States

```mermaid
stateDiagram-v2
  [*] --> Closed
  Closed --> Running: reset() -> adapter.start()
  Running --> Running: step(actions)
  Running --> Running: reset() (close+start new episode)
  Running --> Closed: close() or on_train_end()
  Closed --> [*]
```
