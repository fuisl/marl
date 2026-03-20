from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import wandb as _wandb

    WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    WANDB_AVAILABLE = False


class SafeWandbRun:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled and WANDB_AVAILABLE)
        self._broken = False
        self._finished = False

    @property
    def active(self) -> bool:
        return self.enabled and (not self._broken) and (not self._finished)

    def _mark_broken(self, exc: Exception, where: str) -> None:
        if not self._broken:
            print(f"[wandb] disabled after {where} error: {type(exc).__name__}: {exc}")
        self._broken = True

    def init_training_run(
        self,
        *,
        project: str,
        run_name: str,
        run_config: dict[str, Any] | None,
        out_dir: Path,
        tags: list[str],
        run_metadata: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return

        assert _wandb is not None
        _wandb.init(
            project=project,
            name=run_name,
            config=run_config,
            dir=str(out_dir),
            tags=tags,
            settings=_wandb.Settings(start_method="thread"),
        )
        _wandb.define_metric("Episode")
        for metric_name in (
            "Episode Length",
            "Total Transitions",
            "Elapsed Time (s)",
            "Avg Duration",
            "Avg Waiting Time",
            "Avg Time Loss",
            "Avg Queue Length",
            "Avg Reward",
            "Global Reward",
            "Best Global Reward",
        ):
            _wandb.define_metric(metric_name, step_metric="Episode")

        _wandb.config.update({"run_metadata": dict(run_metadata)}, allow_val_change=True)
        run_obj = _wandb.run
        if run_obj is not None:
            for key, value in run_metadata.items():
                run_obj.summary[str(key)] = value
            run_obj.summary["run_name"] = run_name

    def log(self, payload: Mapping[str, Any]) -> None:
        if not self.active:
            return
        assert _wandb is not None
        try:
            _wandb.log(dict(payload))
        except Exception as exc:  # pragma: no cover - integration/runtime dependent
            self._mark_broken(exc, "log")

    def save(self, path: Path, *, base_path: Path) -> None:
        if not self.active:
            return
        assert _wandb is not None
        try:
            _wandb.save(str(path), base_path=str(base_path))
        except Exception as exc:  # pragma: no cover - integration/runtime dependent
            self._mark_broken(exc, "save")

    def finish(self, *, exit_code: int) -> None:
        if (not self.enabled) or self._finished:
            return
        self._finished = True
        assert _wandb is not None
        try:
            _wandb.finish(exit_code=exit_code)
        except Exception as exc:  # pragma: no cover - integration/runtime dependent
            self._broken = True
            print(f"[wandb] finish warning: {type(exc).__name__}: {exc}")

    def new_table(self, columns: list[str]) -> Any | None:
        if not self.active:
            return None
        assert _wandb is not None
        return _wandb.Table(columns=columns)

    def new_image(self, path: Path) -> Any | None:
        if not self.active:
            return None
        assert _wandb is not None
        return _wandb.Image(str(path))

    def set_summary(self, values: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        assert _wandb is not None
        run_obj = _wandb.run
        if run_obj is None:
            return
        for key, value in values.items():
            run_obj.summary[str(key)] = value
