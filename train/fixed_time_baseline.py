"""Run RESCO-aligned static baselines."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from config_utils import load_dotenv
from marl_env.resco_reporting import to_public_metrics
from marl_env.sumo_env import TrafficSignalEnv
from train.resco_baselines import (
    RescoFixedSignalController,
    maxpressure_actions,
    maxwave_actions,
)
from train.training_logging import (
    TRAIN_LOG_FIELDNAMES,
    build_train_log_row,
    build_train_wandb_payload,
)
from train.wandb_utils import SafeWandbRun

load_dotenv()


def _build_static_actions(
    *,
    policy_name: str,
    env: TrafficSignalEnv,
    fixed_controllers: dict[str, RescoFixedSignalController],
) -> torch.Tensor:
    if policy_name == "FIXED":
        action_map = {signal_id: controller.act() for signal_id, controller in fixed_controllers.items()}
    elif policy_name == "MAXWAVE":
        action_map = maxwave_actions(
            signal_specs=env.get_resco_signal_specs(),
            wave_states=env.get_resco_states("wave"),
        )
    elif policy_name == "MAXPRESSURE":
        action_map = maxpressure_actions(
            signal_specs=env.get_resco_signal_specs(),
            mplight_states=env.get_resco_states("mplight"),
        )
    else:
        raise ValueError(f"Unsupported static RESCO policy {policy_name!r}.")
    return torch.tensor([int(action_map[tl_id]) for tl_id in env.tl_ids], dtype=torch.long)


def run_baseline(
    env_cfg: dict,
    *,
    policy_name: str = "FIXED",
    out_dir: str | Path | None = None,
    wandb_cfg: dict[str, Any] | None = None,
    run_name: str | None = None,
    seed: int = 0,
) -> dict[str, float]:
    output_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_env_cfg = dict(env_cfg)
    resolved_env_cfg.setdefault("benchmark_mode", "resco")
    resolved_env_cfg.setdefault("observation_mode", "graph")
    resolved_env_cfg.setdefault("benchmark_output_dir", str(output_dir))
    if str(resolved_env_cfg.get("benchmark_mode", "native")) != "resco":
        raise ValueError("Static baseline runner requires benchmark_mode='resco'.")

    env = TrafficSignalEnv(**resolved_env_cfg)
    env.reset()
    signal_specs = env.get_resco_signal_specs()
    fixed_controllers = {
        signal_id: RescoFixedSignalController(
            num_actions=len(signal_specs[signal_id]["phase_pairs"]),
            fixed_timings=list(signal_specs[signal_id]["fixed_timings"]),
            fixed_phase_order_idx=int(signal_specs[signal_id]["fixed_phase_order_idx"]),
            fixed_offset=int(signal_specs[signal_id]["fixed_offset"]),
        )
        for signal_id in env.tl_ids
    }

    steps = 0
    t0 = time.perf_counter()
    while True:
        actions = _build_static_actions(
            policy_name=str(policy_name).upper(),
            env=env,
            fixed_controllers=fixed_controllers,
        )
        td = env.step(actions)
        steps += 1
        if td["done"].item():
            break

    elapsed = time.perf_counter() - t0
    raw_metrics = dict(env.get_episode_kpis())
    public_metrics = to_public_metrics(raw_metrics)
    env.close()

    print(f"--- RESCO Static Baseline ({policy_name}) ---")
    for key, value in public_metrics.items():
        print(f"  {key}: {value:.4f}")

    csv_path = output_dir / "train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TRAIN_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            build_train_log_row(
                episode=1,
                n_steps=steps,
                validation_metrics=raw_metrics,
                total_transitions=steps,
                elapsed_s=elapsed,
            )
        )
    print(f"  train_log_csv: {csv_path}")

    wandb_run = SafeWandbRun(
        enabled=bool(isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled", False))
    )
    if wandb_run.enabled:
        wb_name = str(run_name or wandb_cfg.get("run_name") or policy_name)
        wandb_run.init_training_run(
            project=str(wandb_cfg.get("project", "marl-traffic-gat")),
            run_name=wb_name,
            run_config={
                "env": resolved_env_cfg,
                "seed": int(seed),
                "algo": str(policy_name),
            },
            out_dir=output_dir,
            tags=["baseline", "resco", str(policy_name).lower()],
            run_metadata={
                "run_name": wb_name,
                "algo": str(policy_name),
                "seed": int(seed),
            },
        )
        wandb_run.log(
            build_train_wandb_payload(
                episode=1,
                n_steps=steps,
                validation_metrics=raw_metrics,
                total_transitions=steps,
                elapsed_s=elapsed,
                best_global_reward=float(public_metrics["Global Reward"]),
            )
        )
        artifact_paths = env.get_benchmark_artifact_paths()
        for artifact_path in artifact_paths.values():
            artifact = Path(artifact_path)
            if artifact.exists():
                wandb_run.save(artifact, base_path=output_dir)
        wandb_run.set_summary(
            {
                "algo": str(policy_name),
                "Episode Length": float(steps),
                **public_metrics,
            }
        )
        wandb_run.finish(exit_code=0)

    return {key: float(value) for key, value in public_metrics.items()}
