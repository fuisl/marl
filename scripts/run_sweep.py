"""Run Hydra multirun with auto-scaled parallelism from visible GPUs.

Usage:
    python scripts/run_sweep.py scenario=cologne1,cologne8 seed=1,2,3

This script computes the number of visible GPUs (respecting CUDA_VISIBLE_DEVICES),
sets HYDRA_N_JOBS, and launches:
    python scripts/run_experiment.py -m ...
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Sequence

from process_cleanup import terminate_descendants


def _visible_gpu_count() -> int:
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        ids = [x.strip() for x in cuda_visible.split(",") if x.strip() and x.strip() != "-1"]
        return len(ids)

    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _build_command(overrides: Sequence[str], n_jobs: int) -> list[str]:
    return [
        sys.executable,
        "scripts/run_experiment.py",
        "-m",
        f"hydra.launcher.n_jobs={n_jobs}",
        *overrides,
    ]


def _validate_overrides(overrides: Sequence[str]) -> None:
    """Fail fast on common shell paste mistakes.

    A frequent issue is accidentally pasting another full command into the
    override list, which Hydra then parses as an invalid override.
    """
    bad_markers = ("scripts/run_sweep.py", "python ", "python3 ")
    for token in overrides:
        if any(marker in token for marker in bad_markers):
            raise ValueError(
                "Invalid override token detected: "
                f"'{token}'.\n"
                "It looks like a full command was pasted into arguments.\n"
                "Example correct usage:\n"
                "  python scripts/run_sweep.py scenario=cologne1,cologne8 "
                "algo=sac model=gat"
            )


def main() -> int:
    overrides = sys.argv[1:]
    _validate_overrides(overrides)
    gpu_count = _visible_gpu_count()
    n_jobs = gpu_count if gpu_count > 0 else 1

    env = os.environ.copy()
    env["HYDRA_N_JOBS"] = str(n_jobs)

    cmd = _build_command(overrides, n_jobs)
    print(f"[run_sweep] visible_gpus={gpu_count} -> hydra.launcher.n_jobs={n_jobs}")
    print("[run_sweep] command:", " ".join(cmd))

    proc = subprocess.Popen(cmd, env=env)
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        terminate_descendants(proc.pid)
        try:
            proc.send_signal(signal.SIGINT)
            return int(proc.wait(timeout=3.0))
        except subprocess.TimeoutExpired:
            proc.kill()
            return 130
        except ProcessLookupError:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
