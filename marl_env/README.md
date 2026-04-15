# Unified Environment Reference

This repo now uses one environment implementation only: a RESCO-native SUMO
environment exposed by `marl_env.sumo_env.TrafficSignalEnv`.

The goal of this document is to describe the environment contract as it exists
today, so you can build new controllers or RL algorithms directly against it
without having to reverse-engineer the code.

## Scope

- Single public env: `TrafficSignalEnv`
- Single public observation family: canonical per-signal snapshot
- Single public metric pipeline: RESCO-style tripinfo + per-step CSV metrics
- Extensible reward interface: one env, multiple reward functions
- Graph features are derived outside the env by `ObservationAdapter`

Supported local benchmark maps are:

- `grid4x4`
- `arterial4x4`
- `cologne1`
- `cologne3`
- `cologne8`
- `ingolstadt1`
- `ingolstadt7`
- `ingolstadt21`

Unsupported maps fail fast because the env depends on vendored RESCO metadata.

## Design Summary

The environment runs SUMO with RESCO-compatible runtime flags, subscribes to
per-junction vehicle context, builds per-signal rolling observations from those
subscriptions, exposes one canonical flat snapshot per traffic light, and
records episode metrics exactly through:

- `tripinfo_<episode>.xml`
- `metrics_<episode>.csv`

Algorithm-specific views such as `wave`, `mplight`, `drq`, and graph node
features are not emitted by the env. They are derived from the canonical
snapshot plus static signal metadata through `marl_env.observation_adapter`.

## Core Class

```python
from marl_env.sumo_env import TrafficSignalEnv

env = TrafficSignalEnv(
    net_file="nets/grid4x4/grid4x4.net.xml",
    route_file="nets/grid4x4/grid4x4_1.rou.xml",
    step_length=10,
    reward_name="wait",
    max_distance=200,
    output_dir="runs/example",
)
```

### Constructor Arguments

| Argument | Type | Meaning |
|---|---|---|
| `net_file` | `str` | SUMO `.net.xml` path |
| `route_file` | `str` | SUMO route/trip file path |
| `delta_t` | `int` | Decision interval in SUMO seconds |
| `step_length` | `int | None` | Alias for `delta_t`; if provided it overrides `delta_t` |
| `reward_name` | `str` | Reward function key, currently `wait` or `pressure` |
| `yellow_duration` | `int` | Yellow duration used in action transition constraints |
| `all_red_duration` | `int` | All-red clearance duration used in transition constraints |
| `min_green_duration` | `int` | Stored in constraints config, but not currently enforced in the public action mask |
| `sumo_binary` | `str` | Usually `sumo` |
| `gui` | `bool` | Launch with GUI if true |
| `begin_time` | `int` | SUMO begin time |
| `end_time` | `int` | SUMO end time |
| `additional_files` | `list[str] | None` | Extra SUMO input files |
| `timeloss_subscription_policy` | `str` | Time-loss subscription behavior in the TraCI adapter |
| `max_distance` | `int` | RESCO context cutoff distance from the junction |
| `output_dir` | `str | None` | Where raw benchmark artifacts are written |

### Runtime Defaults

From [sumo.yaml](/home/tuancuong/data5t/marl/configs/env/sumo.yaml):

- `step_length = 10`
- `reward_name = wait`
- `max_distance = 200`
- `yellow_duration = 3`
- `all_red_duration = 2`
- `min_green_duration = 10`

## What `reset()` Returns

```python
td = env.reset()
```

Return type: `TensorDict`

Top-level structure:

```text
root
└── agents/
    ├── observation   [n_agents, obs_dim]
    └── action_mask   [n_agents, max_num_actions]
```

The env does not place graph tensors in the returned `TensorDict`.
Static graph information is retrieved separately through `get_graph_metadata()`.

### `td["agents", "observation"]`

- Shape: `[n_agents, obs_dim]`
- Type: `torch.float32`
- Content: canonical flat snapshot for each traffic signal

### `td["agents", "action_mask"]`

- Shape: `[n_agents, max_num_actions]`
- Type: `torch.bool`
- Content: valid local action indices for each signal, padded to the maximum
  action count in the episode

Important: in the current unified env, this mask only removes padded actions
for signals with fewer local phases. It does not currently enforce min-green or
transitional illegality.

## What `step(actions)` Expects And Returns

```python
actions = torch.tensor([...], dtype=torch.long)  # shape [n_agents]
next_td = env.step(actions)
```

### Input

- Shape: `[n_agents]`
- Each entry is a local action index for one traffic light
- Local action indices correspond to the signal's RESCO-compatible green phases

### Output

`step()` returns the same observation structure as `reset()`, plus:

```text
root
├── agents/
│   ├── observation   [n_agents, obs_dim]
│   ├── action_mask   [n_agents, max_num_actions]
│   ├── reward        [n_agents, 1]
│   └── done          [n_agents, 1]
└── done              [1]
```

### Termination

The episode ends when either condition is true:

- `current_time >= end_time`
- `min_expected_vehicles == 0`

## Canonical Observation

The canonical observation is defined by
`marl_env.observation_adapter.CanonicalObservationLayout`.

Feature dimension:

```text
obs_dim = 2 + 5 * max_lanes
```

where `max_lanes` is the maximum number of RESCO-tracked inbound lanes among
all controlled signals in the current scenario.

### Layout

```text
[ phase_index(1) ]
[ phase_length(1) ]
[ lane_mask(max_lanes) ]
[ approaching(max_lanes) ]
[ queued(max_lanes) ]
[ total_wait(max_lanes) ]
[ total_speed(max_lanes) ]
```

### Field Definitions

| Field | Meaning |
|---|---|
| `phase_index` | Current local action index of the active green phase |
| `phase_length` | Time spent in the current green phase, in SUMO seconds |
| `lane_mask` | `1` for real tracked lanes, `0` for padding |
| `approaching` | Vehicles observed on that lane that are not currently queued |
| `queued` | Vehicles on that lane with speed below the queue threshold |
| `total_wait` | Sum of tracked per-vehicle waiting time on that lane |
| `total_speed` | Sum of tracked per-vehicle average speed on that lane |

Padding is always zero-filled.

### Vehicle Tracking Semantics

The env builds the snapshot from rolling `RescoVehicle` objects:

- vehicles are gathered through junction context subscriptions
- only vehicles within `max_distance` of the junction are retained
- queued state is triggered when observed speed is below `0.1`
- `wait` is accumulated while a vehicle remains queued
- speed features use average speed across observations, not only the latest speed

## Static Signal Metadata

Call `env.get_signal_specs()` after `reset()`.

```python
signal_specs = env.get_signal_specs()
spec = signal_specs[env.tl_ids[0]]
```

Each signal spec contains:

| Key | Meaning |
|---|---|
| `signal_id` | Traffic-light id |
| `directions` | RESCO directional keys used by `phase_pairs` |
| `phase_pairs` | Global RESCO phase-pair action list for the map |
| `pair_to_act_map` | Mapping `global_phase_pair_idx -> local_action_idx` |
| `local_num_actions` | Number of local green actions for this signal |
| `fixed_timings` | RESCO fixed-time plan durations |
| `fixed_phase_order_idx` | Fixed-time permutation index |
| `fixed_offset` | Fixed-time offset |
| `lane_order` | Stable lane order used by the canonical snapshot |
| `lane_sets` | Direction -> inbound lane ids |
| `lane_sets_outbound` | Direction -> downstream outbound lane ids |
| `out_lane_to_signal_id` | Outbound lane -> downstream controlled signal |
| `downstream` | Direction -> downstream signal id or `None` |

This is the main source of truth for building controllers outside the env.

## Graph Metadata

Call `env.get_graph_metadata()` after `reset()`.

It returns:

| Field | Meaning |
|---|---|
| `edge_index` | Graph connectivity in COO format |
| `edge_attr` | Edge features, currently `[distance, n_lanes]` |
| `node_ids` | Graph node ids |
| `attached_rl_ids_by_node` | Which RL traffic lights are attached to each graph node |
| `agent_node_indices` | Mapping from RL agents to graph node indices |
| `agent_node_mask` | Mask for valid agent-node mappings |

The env currently builds graph topology with `GraphBuilder(..., mode="all_intersections")`.

## Feature Adapters

Use `ObservationAdapter` to derive algorithm-specific features from the
canonical snapshot.

```python
from marl_env.observation_adapter import ObservationAdapter

adapter = ObservationAdapter(
    signal_specs=env.get_signal_specs(),
    tl_ids=env.tl_ids,
    layout=env.observation_layout,
    graph_metadata=env.get_graph_metadata(),
)
```

### Supported Feature Modes

| Mode | Output |
|---|---|
| `snapshot` | The canonical observation unchanged |
| `wave` | RESCO wave state per signal |
| `mplight` | RESCO MPLight-style pressure state per signal |
| `drq` | RESCO DRQ-style lane-expanded state per signal |

### Main Adapter Methods

#### Per-agent features

```python
agent_x = adapter.agent_features(
    td["agents", "observation"],
    feature_mode="wave",
)
```

- Input shape: `[n_agents, obs_dim]` or `[batch, n_agents, obs_dim]`
- Output shape: `[n_agents, feature_dim]` or `[batch, n_agents, feature_dim]`

#### Graph-node features

```python
graph_x = adapter.graph_features(
    td["agents", "observation"],
    feature_mode="wave",
)
```

- Uses `attached_rl_ids_by_node`
- If a graph node has multiple attached RL signals, features are averaged
- This is what the current graph SAC trainer uses as encoder input

#### Dict form for classical baselines

```python
wave_states = adapter.as_state_dict(
    td["agents", "observation"],
    feature_mode="wave",
)
```

- Output type: `dict[str, list[float]]`
- This is what `MAXWAVE` and `MAXPRESSURE` use

## Reward Interface

The env has one reward hook:

```python
reward_name="wait"      # default
reward_name="pressure"  # available alternate
```

Implemented rewards in [reward.py](/home/tuancuong/data5t/marl/marl_env/reward.py):

### `wait`

```text
reward(signal) = - total_wait(signal)
```

where `total_wait(signal)` is the sum of tracked vehicle waiting time across
that signal's tracked inbound lanes.

### `pressure`

```text
reward(signal) = -(entering_queued - exiting_queued)
```

where `exiting_queued` is measured from downstream tracked lanes based on the
vendored RESCO downstream metadata.

### Extending Rewards

To add a new reward:

1. Add a function `fn(signals: dict[str, RescoSignalState]) -> dict[str, float]`
   in [reward.py](/home/tuancuong/data5t/marl/marl_env/reward.py)
2. Register it in `REWARD_REGISTRY`
3. Pass `reward_name=<new_name>` when constructing the env

This keeps one env and one metric system while allowing experimental rewards.

## Phase And Action Semantics

Actions are local green-phase indices.

On each decision step:

1. The algorithm emits one local action per signal
2. The env maps each local action to a SUMO green phase
3. If the action differs from the current green, the env starts the transition
   via inferred yellow/all-red phases
4. SUMO progresses second by second for `delta_t` seconds
5. The env refreshes tracked vehicle observations and computes rewards

### Important Behavioral Note

The current unified env exposes all local actions as legal in `action_mask`.
That means:

- `action_mask` removes padding only
- min-green is not currently enforced through masking
- if SUMO is already in a transitional phase, new requests are ignored until a
  green phase is active again

If you build a new policy, do not assume `action_mask` encodes min-green.

## Metrics

The env always records RESCO-style raw metrics and exposes them through:

```python
raw_metrics = env.get_episode_metrics()
artifact_paths = env.get_artifact_paths()
```

### Raw Artifacts

Written under `output_dir`:

- `tripinfo_<episode>.xml`
- `metrics_<episode>.csv`

### Raw Metric Keys

`env.get_episode_metrics()` returns keys like:

- `duration`
- `waitingTime`
- `timeLoss`
- `rewards`
- `queue_lengths`
- `max_queues`
- `vehicles`
- `phase_length`
- `global_reward`

### Aggregation Rules

From [resco_reporting.py](/home/tuancuong/data5t/marl/marl_env/resco_reporting.py):

- `duration`, `waitingTime`, `timeLoss` are averaged over trips in `tripinfo`
- `timeLoss` includes `departDelay`
- `rewards`, `queue_lengths`, `max_queues`, `vehicles`, `phase_length` are:
  - averaged across signals per step
  - then averaged across decision steps
- `global_reward` is the episode sum of all local rewards over all signals and steps

### Public Pretty Names

Training and evaluation logs expose:

- `Avg Duration`
- `Avg Waiting Time`
- `Avg Time Loss`
- `Avg Queue Length`
- `Avg Reward`
- `Global Reward`

## How Algorithms Use The Environment

### Graph RL Path

Current local-neighbor graph SAC uses this flow:

1. `env.reset()` returns canonical snapshot observations
2. `ObservationAdapter.graph_features(..., feature_mode=<mode>)` converts them
   into graph node features
3. `env.get_graph_metadata()` provides:
   - `edge_index`
   - `edge_attr`
   - `agent_node_indices`
   - `agent_node_mask`
4. `LocalNeighborGATDiscreteSAC` encodes local and neighbor features, then pools node latents back to agents
5. The actor outputs one local action index per signal

Default adapter mode for the baseline comes from
[default.yaml](/home/tuancuong/data5t/marl/configs/model/default.yaml#L5):

```yaml
observation_adapter:
  feature_mode: snapshot
```

### Static RESCO Controllers

Current classical controllers use:

- `FIXED`
- `STOCHASTIC`
- `MAXWAVE`
- `MAXPRESSURE`

Path:

1. `env.reset()`
2. `signal_specs = env.get_signal_specs()`
3. `adapter = ObservationAdapter(...)`
4. Controller derives `wave` or `mplight` from the canonical snapshot
5. Controller chooses local action indices
6. `env.step(actions)`

Policy-specific defaults are applied in
[fixed_time_baseline.py](/home/tuancuong/data5t/marl/train/fixed_time_baseline.py).

## PettingZoo Wrapper

`marl_env.pettingzoo_env.SumoTrafficParallelEnv` wraps the same core env.

### Reset

```python
observations, infos = env.reset()
```

Per agent:

- `observations[agent]["observation"]`: canonical snapshot vector
- `observations[agent]["action_mask"]`: local padded mask trimmed to that agent

### Step

```python
next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
```

The wrapper:

- converts `dict[str, int]` into the core tensor action format
- returns scalar rewards per agent
- exposes static graph tensors through `get_static_graph_tensors()`

## Practical Examples

### Minimal Core Loop

```python
env = TrafficSignalEnv(...)
td = env.reset()

while True:
    mask = td["agents", "action_mask"]
    actions = torch.zeros(mask.shape[0], dtype=torch.long)
    td = env.step(actions)
    if td["done"].item():
        break

raw_metrics = env.get_episode_metrics()
env.close()
```

### Build Graph Input For A New RL Agent

```python
td = env.reset()
adapter = ObservationAdapter(
    signal_specs=env.get_signal_specs(),
    tl_ids=env.tl_ids,
    layout=env.observation_layout,
    graph_metadata=env.get_graph_metadata(),
)
graph_x = adapter.graph_features(td["agents", "observation"], feature_mode="wave")
graph = env.get_graph_metadata()
```

### Build A Per-signal Controller

```python
td = env.reset()
adapter = ObservationAdapter(
    signal_specs=env.get_signal_specs(),
    tl_ids=env.tl_ids,
    layout=env.observation_layout,
)
wave = adapter.as_state_dict(td["agents", "observation"], feature_mode="wave")
```

## Recommended Integration Rules

If you are adding a new algorithm, treat these as the stable env-facing APIs:

- `reset()`
- `step(actions)`
- `get_signal_specs()`
- `get_graph_metadata()`
- `get_episode_metrics()`
- `get_artifact_paths()`
- `observation_layout`
- `ObservationAdapter`

Try not to depend on private members such as:

- `_signals`
- `_green_phases`
- `_current_green`
- `_yellow_phase_map`

Those are implementation details and may change during cleanup.

## Current Caveats

- `min_green_duration` is configured but not currently enforced in the public
  action mask or action rejection path
- the canonical snapshot is flat; if you need structured tensors, build them
  from `observation_layout`
- the env returns agent observations only; graph tensors come from
  `get_graph_metadata()`
- the repo still contains some older architecture docs that describe the
  previous split env design; this file is the accurate reference for the
  current implementation
