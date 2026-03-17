"""Pytest fixtures and test-environment shims."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
import sys
import types

import pytest


_REAL_TRACI_AVAILABLE = importlib.util.find_spec("traci") is not None
_REAL_LIBSUMO_AVAILABLE = importlib.util.find_spec("libsumo") is not None
_REAL_SUMOLIB_AVAILABLE = importlib.util.find_spec("sumolib") is not None


def _install_stub_module(name: str) -> None:
    """Install a tiny module shim so unit tests import without SUMO."""
    if importlib.util.find_spec(name) is not None or name in sys.modules:
        return

    module = types.ModuleType(name)

    def _unsupported(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(f"{name} is not installed in this test environment.")

    module.start = _unsupported  # type: ignore[attr-defined]
    module.getConnection = _unsupported  # type: ignore[attr-defined]
    module.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None)
    sys.modules[name] = module


_install_stub_module("traci")
_install_stub_module("libsumo")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "sumo: integration tests that require a real SUMO installation.",
    )


@pytest.fixture(scope="session")
def sumo_stack() -> dict[str, object]:
    """Provide SUMO Python tooling for integration tests or skip them."""
    missing: list[str] = []
    if shutil.which("sumo") is None:
        missing.append("sumo binary")
    if shutil.which("netconvert") is None:
        missing.append("netconvert binary")
    if not _REAL_SUMOLIB_AVAILABLE:
        missing.append("sumolib Python package")
    if not _REAL_LIBSUMO_AVAILABLE:
        missing.append("libsumo Python package")

    if missing:
        pytest.skip("SUMO integration tests skipped; missing: " + ", ".join(missing))

    import sumolib  # type: ignore[import-untyped]

    return {"sumolib": sumolib}
