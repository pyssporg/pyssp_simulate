"""Tests for the backend registry and the simulate() facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyssp_simulate import (
    BackendNotAvailableError,
    SimulationSpec,
    available_backends,
    backend_names,
    get_simulator,
    simulate,
)
from pyssp_simulate.interface import BACKEND_PRIORITY


def test_registry_contains_all_backends():
    names = backend_names()
    assert set(names) == {"fmpy", "omsimulator", "ssp4sim", "cosim"}
    assert set(BACKEND_PRIORITY) == set(names)


def test_available_backends_reports_fmpy():
    status = available_backends()
    assert "fmpy" in status
    ok, reason = status["fmpy"]
    assert ok, f"fmpy should be available: {reason}"


def test_available_backends_reports_unavailable_reason():
    status = available_backends()
    for name, (ok, reason) in status.items():
        if not ok:
            assert reason


def test_get_simulator_unknown_name():
    with pytest.raises(KeyError, match="Unknown backend"):
        get_simulator("does-not-exist")


def test_get_simulator_unavailable_backend():
    status = available_backends()
    unavailable = [name for name, (ok, _) in status.items() if not ok]
    if not unavailable:
        pytest.skip("all backends available on this machine")
    with pytest.raises(BackendNotAvailableError):
        get_simulator(unavailable[0])


def test_simulate_auto_uses_fmpy(tmp_path: Path, ssp_archive: Path):
    spec = SimulationSpec(
        ssp=ssp_archive, start_time=0.0, stop_time=0.1, step_size=0.01
    )
    result = simulate(spec, backend="auto")
    # fmpy is a core dependency, so auto always resolves to it at minimum.
    assert result.backend == "fmpy"
    assert "Integrator.y" in result.signals
