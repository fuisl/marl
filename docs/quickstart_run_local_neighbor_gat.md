# Quickstart: Run Local-Neighbor GAT Discrete SAC

This guide shows the shortest path to run experiments with the current baseline method.

## 1) Activate environment

```bash
cd /home/tuancuong/data5t/marl
conda activate marl
```

## 2) Run one training experiment

```bash
python scripts/run_experiment.py
```

This uses defaults from `configs/run.yaml` (local-neighbor GAT + discrete SAC + grid4x4).

## 3) Run a named experiment preset

```bash
python scripts/run_experiment.py experiment=grid5x5
python scripts/run_experiment.py experiment=cologne8
```

## 4) Override scenario/model/algo explicitly

```bash
python scripts/run_experiment.py scenario=grid5x5 algo=sac model=gat seed=21
```

## 5) Run a small sweep (multi-run)

```bash
python scripts/run_experiment.py -m scenario=grid4x4,grid5x5 seed=1,2
```

## 6) Evaluate a saved checkpoint

```bash
python train/evaluate.py checkpoint_path=<PATH_TO_CHECKPOINT>
```

Example:

```bash
python train/evaluate.py checkpoint_path=runs/<run_dir>/checkpoints/best.pt
```

## 7) Visualize graph influence

```bash
python visualization/graph_influence.py checkpoint_path=<PATH_TO_CHECKPOINT>
```

## 8) (Optional) Export topology JSON

```bash
python scripts/export_graph_topology.py --net-file nets/grid5x5/grid5x5.net.xml --out runs/grid5x5_topology.json
```

## Notes

- Outputs are saved under `runs/`.
- Hydra multi-run outputs are saved under `multirun/`.
- If SUMO variables are required in your shell, load them before running.
