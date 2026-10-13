"""Integration tests: every backend runs the generated ramp-to-integrator chain.

The fixture SSP (built at test time around omc-generated FMUs; see
``tests/conftest.py``) connects ``RampSource.y`` to ``Integrator.u``:

    RampSource:  y = t
    Integrator:  der(y) = u, y(0) = 0   =>   y = t²/2

Each backend runs to t = 1 s with step 0.001 s; the recorded signals
must match the analytic solutions, and all backends must agree with
each other within a loose co-simulation tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import (
    START_TIME,
    STEP_SIZE,
    STOP_TIME,
    analytic_integrator,
    analytic_source,
)

from pyssp_simulate import SimulationSpec, available_backends, simulate, simulate_all
from pyssp_simulate.compare import compare
from pyssp_simulate.interface import get_simulator

# fmpy's SSP master records every sample after stepping, i.e. the value
# from t + h lands under the timestamp t, while the other engines record
# aligned samples (see tests/conftest.py). The chain's signals change at
# a rate of at most 1, so a one-step recording offset is bounded by
# STEP_SIZE; the source check therefore needs STEP_SIZE of headroom and
# the integrator's explicit-Euler error (t*h/2 = 5e-4 at t = 1 s)
# dominates its tolerance instead.
SOURCE_ATOL = STEP_SIZE + 1e-6
INTEGRATOR_ATOL = 1e-3
CROSS_BACKEND_ATOL = 5e-3


def _backend_ready(name: str) -> bool:
    ok, _ = available_backends()[name]
    return ok


def assert_matches_analytic(result):
    assert "RampSource.y" in result.signals, result.signals
    assert "Integrator.y" in result.signals

    time = result.time
    source = result["RampSource.y"]
    integrator = result["Integrator.y"]
    assert len(time) > 10
    # Engines record at different offsets (start vs. end of a step), so
    # the first and last samples may sit up to one step outside the
    # nominal window.
    assert time[0] == pytest.approx(START_TIME, abs=STEP_SIZE + 1e-9)
    assert time[-1] == pytest.approx(STOP_TIME, abs=STEP_SIZE + 1e-9)

    np.testing.assert_allclose(source, analytic_source(time), atol=SOURCE_ATOL)
    np.testing.assert_allclose(
        integrator, analytic_integrator(time), atol=INTEGRATOR_ATOL
    )


@pytest.fixture()
def spec(ssp_archive: Path, spec_kwargs) -> SimulationSpec:
    return SimulationSpec(ssp=ssp_archive, **spec_kwargs)


def test_fmpy_backend(spec):
    result = simulate(spec, backend="fmpy")
    assert result.backend == "fmpy"
    assert_matches_analytic(result)
    assert result.runtime_s is not None and result.runtime_s >= 0.0


def test_fmpy_backend_from_unpacked_dir(ssp_dir: Path, spec_kwargs):
    result = simulate(SimulationSpec(ssp=ssp_dir, **spec_kwargs), backend="fmpy")
    assert_matches_analytic(result)


def test_fmpy_backend_writes_result_file(spec, tmp_path: Path):
    output = tmp_path / "results" / "fmpy.csv"
    spec = SimulationSpec(
        ssp=spec.ssp,
        start_time=spec.start_time,
        stop_time=spec.stop_time,
        step_size=spec.step_size,
        result_file=output,
    )
    result = simulate(spec, backend="fmpy")
    assert output.is_file()
    assert output in result.artifacts
    from pyssp_simulate import SimulationResult

    reloaded = SimulationResult.from_csv(output, backend="fmpy")
    np.testing.assert_allclose(reloaded["Integrator.y"], result["Integrator.y"])


@pytest.mark.skipif(not _backend_ready("omsimulator"), reason="OMSimulator not installed")
def test_omsimulator_backend(spec):
    result = simulate(spec, backend="omsimulator")
    assert result.backend == "omsimulator"
    assert_matches_analytic(result)


@pytest.mark.skipif(not _backend_ready("ssp4sim"), reason="pyssp4sim not installed")
def test_ssp4sim_backend(spec):
    result = simulate(spec, backend="ssp4sim")
    assert result.backend == "ssp4sim"
    assert_matches_analytic(result)
    assert any(p.name == "config.json" for p in result.artifacts)


@pytest.mark.skipif(not _backend_ready("cosim"), reason="cosim binary not installed")
def test_cosim_backend(spec):
    result = simulate(spec, backend="cosim")
    assert result.backend == "cosim"
    assert_matches_analytic(result)


def test_simulate_all_runs_available_backends(spec):
    results = simulate_all(spec)
    assert "fmpy" in results
    for backend, result in results.items():
        assert_matches_analytic(result)


@pytest.mark.parametrize(
    "backend",
    [name for name in ("fmpy", "omsimulator", "ssp4sim", "cosim") if _backend_ready(name)],
)
def test_cross_backend_agreement(backend, spec_kwargs, ssp_archive: Path):
    """Every available backend agrees with the FMPy reference result."""
    reference = simulate(SimulationSpec(ssp=ssp_archive, **spec_kwargs), backend="fmpy")
    candidate = simulate(SimulationSpec(ssp=ssp_archive, **spec_kwargs), backend=backend)

    report = compare([reference, candidate], signals=["Integrator.y"])
    assert report.pairs[0].max_abs_error < CROSS_BACKEND_ATOL
    assert report.pairs[0].passed(atol=CROSS_BACKEND_ATOL)


def test_get_simulator_returns_correct_class():
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    assert isinstance(get_simulator("fmpy"), FMPySimulator)
