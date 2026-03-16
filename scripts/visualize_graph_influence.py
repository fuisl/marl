"""Visualize graph topology and City-Networks-style encoder influence."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_utils import load_dotenv, maybe_to_container, resolve_repo_path


load_dotenv()

# Prefer libsumo for visualization runs, matching the training script.
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

from visualization.graph_influence import run_visualization  # noqa: E402


def _default_out_dir(checkpoint_path: Path, env_name: str, method_name: str) -> Path:
    return checkpoint_path.parent / "visualizations" / env_name / method_name


@hydra.main(version_base=None, config_path="../configs", config_name="visualize")
def main(cfg: DictConfig) -> None:
    checkpoint_raw = cfg.runtime.checkpoint_path
    if checkpoint_raw in (None, ""):
        raise ValueError("Set runtime.checkpoint_path in config or via Hydra override.")

    env_common = maybe_to_container(cfg.env.common)
    scenario_params = maybe_to_container(cfg.scenario.env_params)
    if not isinstance(env_common, dict) or not isinstance(scenario_params, dict):
        raise ValueError("Invalid env/scenario config for visualization.")
    env_cfg = {**env_common, **scenario_params}

    checkpoint_path = resolve_repo_path(checkpoint_raw)
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
        model_cfg=maybe_to_container(cfg.model),
        out_dir=out_dir,
        device=str(cfg.runtime.device),
        num_snapshots=int(cfg.analysis.num_snapshots),
        max_hops=maybe_to_container(cfg.analysis.max_hops),
        curve_num_samples=maybe_to_container(cfg.analysis.curve_num_samples),
        map_num_samples=maybe_to_container(cfg.analysis.map_num_samples),
    )

    print("Visualization complete.")
    print(OmegaConf.to_yaml({"artifacts": summary["artifacts"]}, resolve=True).strip())


if __name__ == "__main__":
    main()
