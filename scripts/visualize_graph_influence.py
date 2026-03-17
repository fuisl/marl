"""Visualize graph topology and City-Networks-style encoder influence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_utils import load_dotenv, maybe_to_container, resolve_repo_path


load_dotenv()

from visualization.graph_influence import run_visualization  # noqa: E402


def _default_out_dir(checkpoint_path: Path, env_name: str, method_name: str) -> Path:
    return checkpoint_path.parent / "visualizations" / env_name / method_name


def _load_wandb_config_from_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path | None]:
    wandb_dir = checkpoint_path.parent / "wandb"
    if not wandb_dir.exists():
        return None, None, None

    config_paths = sorted(
        wandb_dir.glob("run-*/files/config.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for config_path in config_paths:
        try:
            wandb_cfg = OmegaConf.load(config_path)
            env_cfg = maybe_to_container(wandb_cfg.env.value)
            model_cfg = maybe_to_container(wandb_cfg.model.value)
        except Exception:
            continue

        if isinstance(env_cfg, dict) and isinstance(model_cfg, dict):
            return env_cfg, model_cfg, config_path

    return None, None, None


@hydra.main(version_base=None, config_path="../configs", config_name="visualize")
def main(cfg: DictConfig) -> None:
    checkpoint_raw = cfg.runtime.checkpoint_path
    if checkpoint_raw in (None, ""):
        raise ValueError("Set runtime.checkpoint_path in config or via Hydra override.")

    checkpoint_path = resolve_repo_path(checkpoint_raw)
    model_cfg = maybe_to_container(cfg.model)

    env_cfg: dict[str, Any]
    if bool(cfg.runtime.use_checkpoint_config):
        checkpoint_env_cfg, checkpoint_model_cfg, source_path = _load_wandb_config_from_checkpoint(
            checkpoint_path
        )
        if checkpoint_env_cfg is not None and checkpoint_model_cfg is not None:
            env_cfg = checkpoint_env_cfg
            model_cfg = checkpoint_model_cfg
            print(f"Loaded env/model config from checkpoint metadata: {source_path}")
        else:
            env_common = maybe_to_container(cfg.env.common)
            scenario_params = maybe_to_container(cfg.scenario.env_params)
            if not isinstance(env_common, dict) or not isinstance(scenario_params, dict):
                raise ValueError("Invalid env/scenario config for visualization.")
            env_cfg = {**env_common, **scenario_params}
            print(
                "Checkpoint metadata not found under wandb; using Hydra env/scenario/model config."
            )
    else:
        env_common = maybe_to_container(cfg.env.common)
        scenario_params = maybe_to_container(cfg.scenario.env_params)
        if not isinstance(env_common, dict) or not isinstance(scenario_params, dict):
            raise ValueError("Invalid env/scenario config for visualization.")
        env_cfg = {**env_common, **scenario_params}

    env_name = Path(str(env_cfg["net_file"])).stem
    method_name = str(env_cfg.get("graph_builder_mode", "original"))

    out_dir_raw = cfg.runtime.out_dir
    if out_dir_raw in (None, ""):
        out_dir = _default_out_dir(checkpoint_path, env_name, method_name)
    else:
        out_dir = resolve_repo_path(out_dir_raw)

    summary = run_visualization(
        checkpoint_path=checkpoint_path,
        env_cfg=env_cfg,
        model_cfg=model_cfg,
        out_dir=out_dir,
        device=str(cfg.runtime.device),
        num_snapshots=int(cfg.analysis.num_snapshots),
        max_hops=maybe_to_container(cfg.analysis.max_hops),
        curve_num_samples=maybe_to_container(cfg.analysis.curve_num_samples),
        map_num_samples=maybe_to_container(cfg.analysis.map_num_samples),
        show_blue_edges_influence_map=bool(cfg.analysis.show_blue_edges_influence_map),
        focal_node_index=maybe_to_container(cfg.analysis.focal_node_index),
        show_heat_contours=bool(cfg.analysis.show_heat_contours),
        num_heat_contours=int(cfg.analysis.num_heat_contours),
        heat_grid_size=int(cfg.analysis.heat_grid_size),
        heat_sigma_scale=float(cfg.analysis.heat_sigma_scale),
        heat_percentile_low=float(cfg.analysis.heat_percentile_low),
        heat_percentile_high=float(cfg.analysis.heat_percentile_high),
    )

    print("Visualization complete.")
    print(OmegaConf.to_yaml({"artifacts": summary["artifacts"]}, resolve=True).strip())


if __name__ == "__main__":
    main()
