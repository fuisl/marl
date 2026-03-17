"""Helpers for reaping child worker processes on interruption.

Hydra multirun with the Joblib launcher uses loky worker processes. On Linux,
Ctrl+C can leave those workers alive after the parent exits. These helpers
enumerate descendant PIDs through ``/proc`` and terminate them explicitly.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path


def _linux_child_pids(pid: int) -> list[int]:
    children_file = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        raw = children_file.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not raw:
        return []
    child_pids: list[int] = []
    for token in raw.split():
        try:
            child_pids.append(int(token))
        except ValueError:
            continue
    return child_pids


def descendant_pids(root_pid: int | None = None) -> list[int]:
    """Return all descendant PIDs for ``root_pid`` on Linux."""
    if os.name != "posix" or not Path("/proc").exists():
        return []

    start_pid = os.getpid() if root_pid is None else int(root_pid)
    seen: set[int] = set()
    descendants: list[int] = []
    stack = [start_pid]

    while stack:
        pid = stack.pop()
        for child_pid in _linux_child_pids(pid):
            if child_pid in seen:
                continue
            seen.add(child_pid)
            descendants.append(child_pid)
            stack.append(child_pid)

    return descendants


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _should_skip_termination(pid: int) -> bool:
    """Return True for descendant processes that should not be force-killed.

    W&B starts an internal service process for logging. Killing it during
    parent cleanup can trigger noisy atexit teardown warnings even when data
    has already been synced.
    """
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    cmdline = raw.replace("\x00", " ").lower()
    wandb_markers = (
        "wandb/sdk/service",
        "wandb service",
        "wandb-service",
        "wandb_internal",
    )
    return any(marker in cmdline for marker in wandb_markers)


def terminate_descendants(
    root_pid: int | None = None,
    *,
    timeout_s: float = 3.0,
) -> list[int]:
    """Terminate all descendants of ``root_pid`` and return their PIDs."""
    descendants = descendant_pids(root_pid)
    if not descendants:
        return []

    pids = [pid for pid in descendants if not _should_skip_termination(pid)]
    if not pids:
        return []

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    deadline = time.monotonic() + max(timeout_s, 0.0)
    alive = {pid for pid in pids if _is_alive(pid)}
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = {pid for pid in alive if _is_alive(pid)}

    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue

    return pids