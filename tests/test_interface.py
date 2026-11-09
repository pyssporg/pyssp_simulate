"""Tests for the backend registry and the simulate() facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyssp_simulate import (
    DEFAULT_BACKEND,
    BackendNotAvailableError,
    SimulationSpec,
    available_backends,
    backend_names,
    get_simulator,
    simulate,
)
from pyssp_simulate.interface import engine_override_path, get_simulator_class


def test_registry_contains_all_backends():
    names = backend_names()
    assert set(names) == {"fmpy", "omsimulator", "ssp4sim", "cosim"}
    assert DEFAULT_BACKEND in names


# --- engine override environment variables ---------------------------------


def test_engine_override_path_accepts_a_package_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    package = tmp_path / "OMSimulator"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.setenv("OMSIMULATOR_PYTHON_PATH", str(tmp_path))

    assert engine_override_path("OMSIMULATOR_PYTHON_PATH", "OMSimulator") == str(
        tmp_path
    )


def test_engine_override_path_unset_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SSP4SIM_PYTHON_PATH", raising=False)

    assert engine_override_path("SSP4SIM_PYTHON_PATH", "pyssp4sim") is None


def test_engine_override_path_missing_module_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SSP4SIM_PYTHON_PATH", str(tmp_path))

    assert engine_override_path("SSP4SIM_PYTHON_PATH", "pyssp4sim") is None


def test_engine_override_path_missing_directory_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("SSP4SIM_PYTHON_PATH", str(tmp_path / "nowhere"))

    assert engine_override_path("SSP4SIM_PYTHON_PATH", "pyssp4sim") is None


def test_engine_override_path_accepts_a_module_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "engine.py").write_text("")
    monkeypatch.setenv("ENGINE_PYTHON_PATH", str(tmp_path))

    assert engine_override_path("ENGINE_PYTHON_PATH", "engine") == str(tmp_path)


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


def test_simulate_defaults_to_the_default_backend(ssp_archive: Path):
    ok, reason = available_backends()[DEFAULT_BACKEND]
    if not ok:
        pytest.skip(f"{DEFAULT_BACKEND} not available: {reason}")
    spec = SimulationSpec(
        ssp=ssp_archive, start_time=0.0, stop_time=0.1, step_size=0.01
    )
    result = simulate(spec)
    assert result.backend == DEFAULT_BACKEND
    assert "Integrator.y" in result.signals


def test_simulate_rejects_unknown_backend(ssp_archive: Path):
    spec = SimulationSpec(
        ssp=ssp_archive, start_time=0.0, stop_time=0.1, step_size=0.01
    )
    with pytest.raises(KeyError, match="Unknown backend"):
        simulate(spec, backend="auto")


def test_simulate_default_backend_unavailable_raises(
    ssp_archive: Path, monkeypatch: pytest.MonkeyPatch
):
    simulator_class = get_simulator_class(DEFAULT_BACKEND)
    monkeypatch.setattr(
        simulator_class,
        "available",
        classmethod(lambda cls: (False, "engine deliberately hidden")),
    )
    spec = SimulationSpec(
        ssp=ssp_archive, start_time=0.0, stop_time=0.1, step_size=0.01
    )
    with pytest.raises(BackendNotAvailableError, match="engine deliberately hidden"):
        simulate(spec)
