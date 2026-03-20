from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from config_utils import maybe_to_container, resolve_repo_path
from train.wandb_utils import SafeWandbRun


_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_config_section(config_name: str, section: str) -> dict[str, Any]:
    cfg_path = _CONFIGS_DIR / config_name
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception:
        return {}

    section_cfg = maybe_to_container(cfg.get(section, {}))
    if isinstance(section_cfg, dict):
        return section_cfg
    return {}


def _eval_metric_stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0.0,
        }

    mean = sum(values) / n
    if n >= 2:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "count": float(n),
    }


def _build_eval_summary_rows(all_metrics: list[dict[str, float]]) -> list[dict[str, float | str]]:
    if not all_metrics:
        return []

    rows: list[dict[str, float | str]] = []
    keys = sorted({k for m in all_metrics for k in m.keys()})
    for key in keys:
        values = [float(m[key]) for m in all_metrics if key in m]
        if not values:
            continue
        stats = _eval_metric_stats(values)
        rows.append(
            {
                "metric": key,
                "mean": stats["mean"],
                "std": stats["std"],
                "mean_pm_std": f"{stats['mean']:.4f} +- {stats['std']:.4f}",
                "min": stats["min"],
                "max": stats["max"],
                "n": stats["count"],
            }
        )
    return rows


def postprocess_after_training(
    cfg: DictConfig,
    *,
    checkpoint_path: Path,
    out_dir: Path,
    env_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    wandb_run: SafeWandbRun,
) -> None:
    post_cfg = maybe_to_container(cfg.train.get("postprocess", {})) or {}
    eval_runtime_defaults = _load_config_section("evaluate.yaml", "runtime")
    viz_runtime_defaults = _load_config_section("visualize.yaml", "runtime")
    viz_analysis_defaults = _load_config_section("visualize.yaml", "analysis")

    run_eval = bool(post_cfg.get("run_evaluation", True))
    run_visual = bool(post_cfg.get("run_visualization", True))

    if not checkpoint_path.exists():
        print(f"[postprocess] skip: checkpoint missing at {checkpoint_path}")
        return

    post_dir = out_dir / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)

    if run_eval:
        try:
            from train.evaluate import evaluate as run_evaluation

            eval_episodes = int(post_cfg.get("eval_episodes", eval_runtime_defaults.get("episodes", 5)))
            eval_gui = bool(post_cfg.get("eval_gui", eval_runtime_defaults.get("gui", False)))
            eval_device = str(post_cfg.get("eval_device", eval_runtime_defaults.get("device", str(device))))
            eval_metrics = run_evaluation(
                checkpoint_path=str(checkpoint_path),
                env_cfg=dict(env_cfg),
                model_cfg=dict(model_cfg),
                n_episodes=eval_episodes,
                device=eval_device,
                gui=eval_gui,
                output_dir=str(post_dir / "eval_raw"),
            )

            eval_file = post_dir / "eval_metrics.json"
            eval_file.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
            print(f"[postprocess] eval metrics saved -> {eval_file}")

            if wandb_run.enabled:
                try:
                    eval_rows = _build_eval_summary_rows(eval_metrics)
                    if eval_rows:
                        table = wandb_run.new_table(
                            ["metric", "mean", "std", "mean_pm_std", "min", "max", "n"]
                        )
                        if table is not None:
                            for row in eval_rows:
                                table.add_data(
                                    row["metric"],
                                    row["mean"],
                                    row["std"],
                                    row["mean_pm_std"],
                                    row["min"],
                                    row["max"],
                                    row["n"],
                                )
                            wandb_run.log({"PostTrain/Eval Summary Table": table})
                    wandb_run.save(eval_file, base_path=out_dir)
                except Exception as exc:  # pragma: no cover - integration/runtime dependent
                    print(f"[postprocess] wandb sync warning (evaluation): {exc}")
        except Exception as exc:  # pragma: no cover - integration/runtime dependent
            print(f"[postprocess] evaluation failed: {exc}")

    if run_visual:
        try:
            from visualization.graph_influence import run_visualization

            viz_out_raw = post_cfg.get("visualization_out_dir", None)
            default_viz_out = viz_runtime_defaults.get("out_dir", None)
            if viz_out_raw:
                viz_out_dir = resolve_repo_path(viz_out_raw)
            elif default_viz_out not in (None, ""):
                viz_out_dir = resolve_repo_path(default_viz_out)
            else:
                viz_out_dir = post_dir / "visualization"

            viz_device = str(
                post_cfg.get(
                    "visualization_device",
                    viz_runtime_defaults.get("device", str(device)),
                )
            )

            num_snapshots = int(
                post_cfg.get(
                    "visualization_num_snapshots",
                    viz_analysis_defaults.get("num_snapshots", 5),
                )
            )
            max_hops = maybe_to_container(
                post_cfg.get(
                    "visualization_max_hops",
                    viz_analysis_defaults.get("max_hops", None),
                )
            )
            curve_num_samples = maybe_to_container(
                post_cfg.get(
                    "visualization_curve_num_samples",
                    viz_analysis_defaults.get("curve_num_samples", None),
                )
            )
            map_num_samples = maybe_to_container(
                post_cfg.get(
                    "visualization_map_num_samples",
                    viz_analysis_defaults.get("map_num_samples", None),
                )
            )
            show_blue_edges = bool(
                post_cfg.get(
                    "visualization_show_blue_edges_influence_map",
                    viz_analysis_defaults.get("show_blue_edges_influence_map", False),
                )
            )
            focal_node_index = maybe_to_container(
                post_cfg.get(
                    "visualization_focal_node_index",
                    viz_analysis_defaults.get("focal_node_index", None),
                )
            )
            show_heat_contours = bool(
                post_cfg.get(
                    "visualization_show_heat_contours",
                    viz_analysis_defaults.get("show_heat_contours", True),
                )
            )
            num_heat_contours = int(
                post_cfg.get(
                    "visualization_num_heat_contours",
                    viz_analysis_defaults.get("num_heat_contours", 6),
                )
            )
            heat_grid_size = int(
                post_cfg.get(
                    "visualization_heat_grid_size",
                    viz_analysis_defaults.get("heat_grid_size", 220),
                )
            )
            heat_sigma_scale = float(
                post_cfg.get(
                    "visualization_heat_sigma_scale",
                    viz_analysis_defaults.get("heat_sigma_scale", 0.06),
                )
            )
            heat_percentile_low = float(
                post_cfg.get(
                    "visualization_heat_percentile_low",
                    viz_analysis_defaults.get("heat_percentile_low", 2.0),
                )
            )
            heat_percentile_high = float(
                post_cfg.get(
                    "visualization_heat_percentile_high",
                    viz_analysis_defaults.get("heat_percentile_high", 98.0),
                )
            )
            heat_weight_mode = str(
                post_cfg.get(
                    "visualization_heat_weight_mode",
                    viz_analysis_defaults.get("heat_weight_mode", "raw"),
                )
            )

            vis_summary = run_visualization(
                checkpoint_path=checkpoint_path,
                env_cfg=dict(env_cfg),
                model_cfg=dict(model_cfg),
                out_dir=viz_out_dir,
                device=viz_device,
                num_snapshots=num_snapshots,
                max_hops=max_hops,
                curve_num_samples=curve_num_samples,
                map_num_samples=map_num_samples,
                show_blue_edges_influence_map=show_blue_edges,
                focal_node_index=focal_node_index,
                show_heat_contours=show_heat_contours,
                num_heat_contours=num_heat_contours,
                heat_grid_size=heat_grid_size,
                heat_sigma_scale=heat_sigma_scale,
                heat_percentile_low=heat_percentile_low,
                heat_percentile_high=heat_percentile_high,
                heat_weight_mode=heat_weight_mode,
            )

            print(f"[postprocess] visualization saved -> {viz_out_dir}")

            if wandb_run.enabled:
                try:
                    artifacts = vis_summary.get("artifacts", {})
                    if isinstance(artifacts, dict):
                        graph_path = Path(str(artifacts.get("graph_topology", "")))
                        curve_path = Path(str(artifacts.get("influence_curve", "")))
                        map_path = Path(str(artifacts.get("influence_map", "")))
                        image_payload: dict[str, Any] = {}
                        graph_image = wandb_run.new_image(graph_path) if graph_path.exists() else None
                        curve_image = wandb_run.new_image(curve_path) if curve_path.exists() else None
                        map_image = wandb_run.new_image(map_path) if map_path.exists() else None
                        if graph_image is not None:
                            image_payload["PostTrain/Visualization/Graph Topology"] = graph_image
                        if curve_image is not None:
                            image_payload["PostTrain/Visualization/Influence Curve"] = curve_image
                        if map_image is not None:
                            image_payload["PostTrain/Visualization/Influence Map"] = map_image
                        if image_payload:
                            wandb_run.log(image_payload)
                        for artifact_path in artifacts.values():
                            artifact_file = Path(str(artifact_path))
                            if artifact_file.exists():
                                wandb_run.save(artifact_file, base_path=out_dir)
                except Exception as exc:  # pragma: no cover - integration/runtime dependent
                    print(f"[postprocess] wandb sync warning (visualization): {exc}")
        except Exception as exc:  # pragma: no cover - integration/runtime dependent
            print(f"[postprocess] visualization failed: {exc}")
