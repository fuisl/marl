## PettingZoo Environment — Input / Output Reference

### `reset() → (observations, infos)`

```python
observations, infos = env.reset()
```

| Field | Type | Shape | Description |
|---|---|---|---|
| `observations[agent]["observation"]` | `np.float32` | `(4·max_lanes + max_green + 1,)` | Per-agent state vector |
| `observations[agent]["action_mask"]` | `np.int8` | `(n_i,)` | Legal action mask; `1` = legal |
| `infos[agent]` | `dict` | — | Always `{}` (empty) |

---

### `step(actions) → (obs, rewards, terminations, truncations, infos)`

**Input:**
```python
actions: dict[str, int]   # agent_id → action index into green phases
```

| Value | Meaning |
|---|---|
| `0` | Stay on / switch to green phase 0 |
| `1` | Switch to green phase 1 |
| … | … up to `n_i - 1` for agent `i` |

**Outputs:**

| Return | Type | Shape per agent | Description |
|---|---|---|---|
| `observations` | `dict[str, dict]` | same as reset | Post-step observation |
| `rewards` | `dict[str, float]` | scalar | Reward (see below) |
| `terminations` | `dict[str, bool]` | bool | `True` if sim ended (time or vehicles exhausted) |
| `truncations` | `dict[str, bool]` | bool | Always `False` (not used) |
| `infos` | `dict[str, dict]` | `{}` | Empty |

---

### Observation Vector Layout

Shape: `(4·max_lanes + max_green + 1,)` — same dimension for all agents (padded)

```
[ queue(0..max_lanes-1)       ]  # halting vehicles per lane
[ wait(0..max_lanes-1)        ]  # cumulative waiting time per lane (s)
[ occupancy(0..max_lanes-1)   ]  # lane occupancy [0,1]
[ speed(0..max_lanes-1)       ]  # mean speed (m/s)
[ phase_onehot(0..max_green-1)]  # one-hot of current green phase index
[ elapsed(1)                  ]  # seconds held in current green phase
```

For a 4×4 grid with homogeneous intersections (e.g. 4 controlled lanes, 2 green phases):
- `max_lanes = 4`, `max_green = 2` → `obs_dim = 4·4 + 2 + 1 = 19`

Agents with fewer lanes than `max_lanes` have trailing zeros.

---

### Action Space

```python
env.action_space("J0")  # → Discrete(n_i)
```

- `n_i` = number of green phases for agent `i` (from SUMO TLS program)
- Action `k` means "I want to hold/switch to green phase index `k`"
- The env enforces `min_green_duration` and handles yellow/all-red transitions internally
- `action_mask[k] == 0` means that action is currently illegal (in transition, or min-green not satisfied)

---

### Reward (per agent, per step)

**Combined mode** (default):
$$r = w_{queue} \cdot \sum \text{queue} + w_{wait} \cdot \sum \text{wait} + w_{speed} \cdot \bar{v} + w_{throughput} \cdot N_{cleared}$$

Default weights from env.yaml:

| Component | Weight | Direction |
|---|---|---|
| queue (halting vehicles) | −0.25 | minimize |
| wait (cumulative wait, s) | −0.25 | minimize |
| mean speed (m/s) | +0.25 | maximize |
| throughput (vehicles cleared) | +0.25 | maximize |

Other modes: `"queue"` (`-Σqueue`), `"wait"` (`-Σwait`), `"pressure"` (`-Σqueue`).

---

### Static Graph Tensors (for GNN policies)

Available after `reset()`:
```python
env.edge_index   # torch.Tensor [2, E]  — COO adjacency
env.edge_attr    # torch.Tensor [E, 2]  — [distance_m, n_lanes]
```

Nodes = traffic-light agents (stable integer index = `env.get_agent_index_map()`).  
Graph topology is **fixed** for the entire episode.

---

### Key Configuration Knobs

| Parameter | Default | Effect |
|---|---|---|
| `delta_t` | 5 s | Seconds of simulation per RL step |
| `min_green_duration` | 5 s | Min ticks before phase switch is legal |
| `yellow_duration` | 3 s | Yellow phase length (SUMO-managed) |
| `all_red_duration` | 1 s | All-red clearance length (SUMO-managed) |
| `end_time` | 3600 s | Episode length |
| `illegal_action_mode` | `"coerce"` | `"coerce"` / `"raise"` / `"penalize"` |

---

### State Design Checklist

For a GNN-based policy:

- **Node features**: `observation` vector `(obs_dim,)` per agent — contains local lane stats + current phase + elapsed time
- **Edge features**: `edge_attr` `[E, 2]` — static distance + lane count between intersections
- **Graph structure**: `edge_index` `[2, E]` — fixed COO adjacency
- **Action masking**: `action_mask` `(n_i,)` — must be applied before sampling (mask out logits with `−inf`)
- **Heterogeneous agents**: `n_i` differs per agent; use per-agent heads or pad + mask