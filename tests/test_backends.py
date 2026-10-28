"""Integration tests: every backend runs the generated ramp-to-integrator chain.

The fixture SSP (built at test time around omc-generated FMUs; see
``tests/conftest.py``) connects ``RampSource.y`` to ``Integrator.u``:

    RampSource:  y = t
    Integrator:  der(y) = u, y(0) = 0   =>   y = t²/2

Each backend runs to t = 1 s with step 0.001 s; the recorded signals
must match the analytic solutions, and all backends must agree with
each other within a loose co-simulation tolerance.

The adapter tests at the end stub the engines, so they run without any
engine installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pyssp_vv.measure import Trajectory, compare
from tests.conftest import (
    CONTAINER_NAMES,
    START_TIME,
    STEP_COUNT,
    STEP_SIZE,
    analytic_integrator,
    analytic_source,
)

from pyssp_simulate import (
    SimulationSpec,
    available_backends,
    backend_names,
    simulate,
    simulate_all,
)
from pyssp_simulate.backends import ssp4sim_backend
from pyssp_simulate.backends.cosim_backend import (
    CosimSimulator,
    component_name_from_csv,
    merge_cosim_csvs,
    resolve_cosim_binary,
)
from pyssp_simulate.backends.fmpy_backend import _canonicalize, _get_connections
from pyssp_simulate.backends.omsimulator_backend import OMSimulatorSimulator
from pyssp_simulate.backends.ssp4sim_backend import (
    DIAGNOSTIC_ARTIFACTS,
    SSP4SimSimulator,
)
from pyssp_simulate.interface import get_simulator
from pyssp_simulate.result import read_result_csv

# The recorded OMSimulator result excerpt: the real engine's header for this
# fixture package plus its first data rows (see tests/fixtures/README.md).
OMSIMULATOR_FIXTURE = Path(__file__).parent / "fixtures" / "omsimulator_result.csv"

# Documented recording policy of each backend for the fixture window. The
# fixture has STEP_COUNT = stop/step = 1000 communication steps.
#
# * fmpy records each sample after stepping, so the value of step k lands
#   under the step's start timestamp: indices 0 .. STEP_COUNT - 1, last
#   timestamp STOP - STEP.
# * ssp4sim records start-of-step values: indices 1 .. STEP_COUNT, first
#   timestamp START + STEP, last timestamp STOP.
# * OMSimulator logs every step boundary including both ends: indices
#   0 .. STEP_COUNT (with event output disabled, see ``spec_kwargs``).
# * cosim (libcosim) logs every communication step over [START, STOP]:
#   indices 0 .. STEP_COUNT. Unverified here: the binary is not installed.
RECORDING_FIRST_INDEX = {"fmpy": 0, "ssp4sim": 1, "omsimulator": 0, "cosim": 0}
RECORDING_SAMPLE_COUNT = {
    "fmpy": STEP_COUNT,
    "ssp4sim": STEP_COUNT,
    "omsimulator": STEP_COUNT + 1,
    "cosim": STEP_COUNT + 1,
}

# fmpy's one-step recording offset is bounded by STEP_SIZE; the source check
# therefore needs STEP_SIZE of headroom and the integrator's explicit-Euler
# error (t*h/2 = 5e-4 at t = 1 s) dominates its tolerance instead.
SOURCE_ATOL = STEP_SIZE + 1e-6
INTEGRATOR_ATOL = 1e-3
CROSS_BACKEND_ATOL = 5e-3


def _backend_ready(name: str) -> bool:
    ok, _ = available_backends()[name]
    return ok


def assert_sample_grid(result) -> None:
    """Assert the documented, engine-exact time grid for the fixture.

    Exactly the documented number of samples, strictly increasing, no
    duplicate timestamps, and every sample on the ``start + k * step`` grid
    from the backend's documented first index.
    """
    assert result.backend in RECORDING_FIRST_INDEX, result.backend
    first = RECORDING_FIRST_INDEX[result.backend]
    count = RECORDING_SAMPLE_COUNT[result.backend]
    time = result.time

    assert len(time) == count, (result.backend, len(time), count)
    assert np.all(np.diff(time) > 0.0), f"{result.backend} time is not strictly increasing"
    assert len(np.unique(time)) == len(time), f"{result.backend} time has duplicates"
    expected = START_TIME + np.arange(first, first + count) * STEP_SIZE
    np.testing.assert_allclose(time, expected, atol=1e-9, rtol=0)


def assert_matches_analytic(result):
    assert "RampSource.y" in result.signals, result.signals
    assert "Integrator.y" in result.signals

    assert_sample_grid(result)
    time = result.time
    source = result["RampSource.y"]
    integrator = result["Integrator.y"]

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

    report = compare(
        Trajectory(reference.time, {name: reference[name] for name in reference.signals}),
        Trajectory(candidate.time, {name: candidate[name] for name in candidate.signals}),
        signals=["Integrator.y"],
    )
    assert report.signals[0].max_abs_error < CROSS_BACKEND_ATOL


def test_common_columns_have_identical_order(spec):
    """Shared columns line up across engines: ``time`` first, then sorted."""
    ready = [name for name in backend_names() if _backend_ready(name)]
    if not ready:
        pytest.skip("no simulation backend available")
    results = {name: simulate(spec, backend=name) for name in ready}

    common = set.intersection(
        *(set(result.data.dtype.names or ()) for result in results.values())
    )
    common_orders = {}
    for name, result in results.items():
        names = tuple(result.data.dtype.names or ())
        assert names[0] == "time", (name, names)
        assert tuple(sorted(result.signals)) == result.signals, (name, names)
        common_orders[name] = tuple(field for field in names if field in common)

    assert len(set(common_orders.values())) <= 1, common_orders


@pytest.mark.level_contract
def test_get_simulator_returns_correct_class():
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    assert isinstance(get_simulator("fmpy"), FMPySimulator)


# --- adapter unit tests (stubbed engines) -----------------------------------

_SSD_HEADER = (
    '<?xml version="1.0"?>'
    '<ssd:SystemStructureDescription'
    ' xmlns:ssd="http://ssp-standard.org/SSP1/SystemStructureDescription"'
    ' xmlns:ssv="http://ssp-standard.org/SSP1/SystemStructureParameterValues"'
    ' xmlns:ssm="http://ssp-standard.org/SSP1/SystemStructureParameterMapping"'
    ' xmlns:ssc="http://ssp-standard.org/SSP1/SystemStructureCommon"'
    ' name="Synthetic" version="2.0">'
)


def _synthetic_ssp_dir(
    tmp_path: Path,
    *,
    binding: str = "",
    resources: dict[str, str] | None = None,
) -> Path:
    """An unpacked SSP tree with one unpacked FMU and an optional binding."""
    root = tmp_path / "Synthetic"
    fmu_dir = root / "resources" / "Source"
    (fmu_dir / "binaries" / "linux64").mkdir(parents=True)
    (fmu_dir / "modelDescription.xml").write_text("<fmiModelDescription/>")
    (fmu_dir / "binaries" / "linux64" / "model.so").write_bytes(b"\x7fELF")
    for name, text in (resources or {}).items():
        resource = root / name
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_text(text)
    (root / "SystemStructure.ssd").write_text(
        _SSD_HEADER + f'<ssd:System name="system">{binding}</ssd:System>'
        "</ssd:SystemStructureDescription>"
    )
    return root


def _inline_binding(values: dict[str, float], mapping: str | None = None) -> str:
    parameters = "".join(
        f'<ssv:Parameter name="{name}"><ssv:Real value="{value}"/></ssv:Parameter>'
        for name, value in values.items()
    )
    mapping_xml = (
        f'<ssd:ParameterMapping source="{mapping}"/>' if mapping is not None else ""
    )
    return (
        "<ssd:ParameterBindings><ssd:ParameterBinding>"
        "<ssd:ParameterValues>"
        '<ssv:ParameterSet version="1.0" name="inline">'
        f"<ssv:Parameters>{parameters}</ssv:Parameters>"
        "</ssv:ParameterSet>"
        "</ssd:ParameterValues>"
        f"{mapping_xml}"
        "</ssd:ParameterBinding></ssd:ParameterBindings>"
    )


def _ssm(*entries: tuple[str, str]) -> str:
    body = "".join(
        f'<ssm:MappingEntry source="{source}" target="{target}"/>'
        for source, target in entries
    )
    return (
        '<ssm:ParameterMapping'
        ' xmlns:ssm="http://ssp-standard.org/SSP1/SystemStructureParameterMapping">'
        f"{body}</ssm:ParameterMapping>"
    )


def _install_ssp4sim_stub(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the child process: record the config and fabricate its output."""
    observed: dict = {}

    def fake_run(command, capture_output=True, text=True, timeout=None):
        config = json.loads(Path(command[-1]).read_text())
        simulation = config["simulation"]
        runtime_root = Path(simulation["ssp"])
        observed["runtime_root"] = runtime_root
        observed["unpacked_fmu"] = (
            runtime_root / "resources" / "Source" / "modelDescription.xml"
        ).is_file()
        observed["tolerance"] = simulation.get("tolerance")
        Path(simulation["recording"]["result_file"]).write_text(
            "time,Source.y\n0.0,0.0\n1.0,1.0\n"
        )
        for name in DIAGNOSTIC_ARTIFACTS:
            (Path(simulation["working_dir"]) / name).write_text("diagnostic")
        return SimpleNamespace(returncode=0, stderr="")

    from pyssp_simulate.backends import ssp4sim_backend

    monkeypatch.setattr(ssp4sim_backend.subprocess, "run", fake_run)
    return observed


@pytest.mark.level_contract
def test_ssp4sim_uses_unpacked_fmus_and_cleans_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ssp = _synthetic_ssp_dir(tmp_path)
    work = tmp_path / "work"
    observed = _install_ssp4sim_stub(monkeypatch)

    result = SSP4SimSimulator().simulate(
        SimulationSpec(
            ssp=ssp,
            start_time=0.0,
            stop_time=1.0,
            step_size=0.1,
            tolerance=1e-3,
            working_dir=work,
        )
    )

    assert observed["unpacked_fmu"] is True
    # The caller's tolerance is passed through verbatim, never substituted.
    assert observed["tolerance"] == 1e-3
    assert not (observed["runtime_root"] / "resources" / "Source.fmu").exists()
    for name in DIAGNOSTIC_ARTIFACTS:
        assert not (work / name).exists()
    assert result.backend == "ssp4sim"
    assert any(path.name == "config.json" for path in result.artifacts)


@pytest.mark.level_contract
def test_ssp4sim_requires_an_explicit_tolerance(tmp_path: Path):
    """No silent default: the engine requires the key, so the caller must pass it."""
    ssp = _synthetic_ssp_dir(tmp_path)

    with pytest.raises(ValueError, match="requires an explicit tolerance"):
        SSP4SimSimulator().simulate(
            SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0, step_size=0.1)
        )


def _ssp4sim_spec(ssp: Path, **kwargs) -> SimulationSpec:
    return SimulationSpec(
        ssp=ssp, start_time=0.0, stop_time=1.0, step_size=0.1, tolerance=1e-3, **kwargs
    )


def _write_fake_ssp4sim_child(tmp_path: Path, body: str) -> Path:
    """A child script in ``tmp_path``; the real engine is never imported."""
    script = tmp_path / "fake_ssp4sim_child.py"
    script.write_text(body)
    return script


@pytest.mark.level_contract
def test_ssp4sim_child_nonzero_exit_raises(tmp_path: Path, monkeypatch):
    """A crashed child must surface as an error, never as a silent pass."""
    ssp = _synthetic_ssp_dir(tmp_path)
    child = _write_fake_ssp4sim_child(
        tmp_path, "import sys\nsys.stderr.write('engine exploded\\n')\nsys.exit(3)\n"
    )
    monkeypatch.setattr(ssp4sim_backend, "_CHILD_SCRIPT", child)

    with pytest.raises(RuntimeError, match="exit code 3") as error:
        SSP4SimSimulator().simulate(_ssp4sim_spec(ssp))
    assert "engine exploded" in str(error.value)


@pytest.mark.level_contract
def test_ssp4sim_child_without_result_file_raises(tmp_path: Path, monkeypatch):
    """Exit code 0 is not success unless the result file exists."""
    ssp = _synthetic_ssp_dir(tmp_path)
    child = _write_fake_ssp4sim_child(tmp_path, "import sys\nsys.exit(0)\n")
    monkeypatch.setattr(ssp4sim_backend, "_CHILD_SCRIPT", child)

    with pytest.raises(RuntimeError, match="did not produce a result file"):
        SSP4SimSimulator().simulate(_ssp4sim_spec(ssp))


@pytest.mark.level_contract
def test_ssp4sim_hung_child_times_out(tmp_path: Path, monkeypatch):
    """A hung engine raises a bounded timeout instead of hanging CI."""
    ssp = _synthetic_ssp_dir(tmp_path)
    child = _write_fake_ssp4sim_child(tmp_path, "import time\ntime.sleep(30)\n")
    monkeypatch.setattr(ssp4sim_backend, "_CHILD_SCRIPT", child)

    with pytest.raises(TimeoutError, match="did not finish within 0.2 s"):
        SSP4SimSimulator(timeout=0.2).simulate(_ssp4sim_spec(ssp))


class _FakeOMInstance:
    def __init__(self, calls: dict, *, write_result: bool = True):
        self.calls = calls
        self.write_result = write_result
        self.result_path: Path | None = None

    def setStartTime(self, value):
        self.calls["start_time"].append(value)

    def setStopTime(self, value):
        self.calls["stop_time"].append(value)

    def setLoggingInterval(self, value):
        self.calls["logging"].append(value)

    def setFixedStepSize(self, value):
        self.calls["fixed"].append(value)

    def setTolerance(self, value):
        self.calls["tolerance"].append(value)

    def setResultFile(self, path):
        self.result_path = Path(path)

    def initialize(self):
        pass

    def simulate(self):
        assert self.result_path is not None
        if self.write_result:
            # Write the recorded real-engine output shape, not a simplified
            # ``time,Source.y`` header the engine never emits.
            self.result_path.write_text(OMSIMULATOR_FIXTURE.read_text())

    def terminate(self):
        pass

    def delete(self):
        pass


def _install_omsimulator_stub(
    monkeypatch: pytest.MonkeyPatch, *, write_result: bool = True
) -> dict:
    calls: dict = {"options": [], "logging": [], "fixed": [], "start_time": [], "stop_time": []}
    status = SimpleNamespace(ok="ok")

    class FakeCapi:
        @staticmethod
        def setCommandLineOption(option):
            calls["options"].append(option)
            return status.ok if not option.startswith("--bad") else "error"

    class FakeModel:
        def __init__(self, archive):
            self.archive = archive

        def instantiate(self):
            return _FakeOMInstance(calls, write_result=write_result)

    fake_capi = ModuleType("OMSimulator.capi")
    fake_capi.Capi = FakeCapi
    fake_capi.Status = status
    fake_module = ModuleType("OMSimulator")
    fake_module.SSP = FakeModel
    fake_module.Settings = SimpleNamespace()
    fake_module.capi = fake_capi

    monkeypatch.setitem(sys.modules, "OMSimulator", fake_module)
    monkeypatch.setitem(sys.modules, "OMSimulator.capi", fake_capi)
    return calls


def _omsimulator_spec(ssp: Path, **kwargs) -> SimulationSpec:
    return SimulationSpec(
        ssp=ssp, start_time=0.0, stop_time=1.0, step_size=0.1, **kwargs
    )


@pytest.mark.level_contract
def test_omsimulator_fixed_step_false_sets_only_logging_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ssp = _synthetic_ssp_dir(tmp_path)
    calls = _install_omsimulator_stub(monkeypatch)

    result = OMSimulatorSimulator().simulate(_omsimulator_spec(ssp, fixed_step=False))

    assert calls["logging"] == [pytest.approx(0.1)]
    assert calls["fixed"] == []
    assert calls["options"] == []
    assert result.backend == "omsimulator"


@pytest.mark.level_contract
def test_omsimulator_fixed_step_true_sets_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ssp = _synthetic_ssp_dir(tmp_path)
    calls = _install_omsimulator_stub(monkeypatch)

    OMSimulatorSimulator().simulate(_omsimulator_spec(ssp))

    assert calls["logging"] == [pytest.approx(0.1)]
    assert calls["fixed"] == [pytest.approx(0.1)]


@pytest.mark.level_contract
def test_omsimulator_options_reach_the_engine_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ssp = _synthetic_ssp_dir(tmp_path)
    calls = _install_omsimulator_stub(monkeypatch)
    options = ("--wallTime=true", "--emitEvents=false")

    OMSimulatorSimulator().simulate(_omsimulator_spec(ssp, omsimulator_options=options))

    assert calls["options"] == list(options)


@pytest.mark.level_contract
def test_omsimulator_rejected_option_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ssp = _synthetic_ssp_dir(tmp_path)
    _install_omsimulator_stub(monkeypatch)

    with pytest.raises(RuntimeError, match="command line option"):
        OMSimulatorSimulator().simulate(
            _omsimulator_spec(ssp, omsimulator_options=("--bad=true",))
        )


@pytest.mark.level_contract
def test_omsimulator_missing_result_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An engine run that writes no CSV must fail, not return empty data."""
    ssp = _synthetic_ssp_dir(tmp_path)
    _install_omsimulator_stub(monkeypatch, write_result=False)

    with pytest.raises(RuntimeError, match="did not produce a result file"):
        OMSimulatorSimulator().simulate(_omsimulator_spec(ssp))


def _recorded_omsimulator_columns() -> tuple[str, ...]:
    return read_result_csv(
        OMSIMULATOR_FIXTURE, root_system=CONTAINER_NAMES
    ).dtype.names


@pytest.mark.level_contract
def test_omsimulator_stub_matches_recorded_header(
    ssp_archive: Path, monkeypatch: pytest.MonkeyPatch
):
    """The fake engine emits the recorded real header for the same SSP."""
    _install_omsimulator_stub(monkeypatch)

    result = OMSimulatorSimulator().simulate(
        SimulationSpec(ssp=ssp_archive, start_time=0.0, stop_time=1.0, step_size=0.1)
    )

    assert result.data.dtype.names == _recorded_omsimulator_columns()


@pytest.mark.skipif(not _backend_ready("omsimulator"), reason="OMSimulator not installed")
def test_omsimulator_real_backend_matches_recorded_header(spec):
    """The real engine still produces the recorded canonical column set."""
    result = simulate(spec, backend="omsimulator")

    assert result.data.dtype.names == _recorded_omsimulator_columns()


def _install_fmpy_stub(monkeypatch: pytest.MonkeyPatch) -> dict:
    from fmpy.ssp import simulation as fmpy_simulation

    observed: dict = {}

    def fake_simulate_ssp(
        archive, start_time=0.0, stop_time=1.0, step_size=None, parameter_set=None
    ):
        observed["archive"] = archive
        observed["parameter_set"] = parameter_set
        return np.array(
            [(0.0, 0.0), (1.0, 1.0)],
            dtype=[("time", np.float64), ("Source.y", np.float64)],
        )

    monkeypatch.setattr(fmpy_simulation, "simulate_ssp", fake_simulate_ssp)
    return observed


def _fmpy_spec(ssp: Path, *, validate_ssd: bool = False) -> SimulationSpec:
    return SimulationSpec(
        ssp=ssp,
        start_time=0.0,
        stop_time=1.0,
        step_size=0.1,
        validate_ssd=validate_ssd,
    )


def _parameter_names(parameter_set) -> list[str]:
    return [parameter.name for parameter in parameter_set.parameters]


@pytest.mark.level_contract
def test_fmpy_passes_the_declared_parameter_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(tmp_path, binding=_inline_binding({"height": 2.0}))
    observed = _install_fmpy_stub(monkeypatch)

    FMPySimulator().simulate(_fmpy_spec(ssp))

    assert _parameter_names(observed["parameter_set"]) == ["height"]


@pytest.mark.level_contract
def test_fmpy_applies_the_ssm_mapping_to_target_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(
        tmp_path,
        binding=_inline_binding({"height": 2.0}, mapping="resources/step.ssm"),
        resources={"resources/step.ssm": _ssm(("height", "step.height"))},
    )
    observed = _install_fmpy_stub(monkeypatch)

    FMPySimulator().simulate(_fmpy_spec(ssp))

    assert _parameter_names(observed["parameter_set"]) == ["step.height"]


@pytest.mark.level_contract
def test_fmpy_mapping_source_absent_from_the_set_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(
        tmp_path,
        binding=_inline_binding({"height": 2.0}, mapping="resources/step.ssm"),
        resources={"resources/step.ssm": _ssm(("width", "step.width"))},
    )
    _install_fmpy_stub(monkeypatch)

    with pytest.raises(ValueError, match="width"):
        FMPySimulator().simulate(_fmpy_spec(ssp))


@pytest.mark.level_contract
def test_fmpy_without_parameter_binding_passes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(tmp_path)
    observed = _install_fmpy_stub(monkeypatch)

    FMPySimulator().simulate(_fmpy_spec(ssp))

    assert observed["parameter_set"] is None


@pytest.mark.level_contract
def test_fmpy_validate_ssd_false_skips_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fmpy.ssp import ssd as fmpy_ssd

    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(tmp_path)
    _install_fmpy_stub(monkeypatch)
    validated: list = []
    monkeypatch.setattr(
        fmpy_ssd, "validate_tree", lambda *args, **kwargs: validated.append(args)
    )

    FMPySimulator().simulate(_fmpy_spec(ssp))

    assert validated == []


@pytest.mark.level_contract
def test_fmpy_validate_ssd_true_keeps_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fmpy.ssp import ssd as fmpy_ssd

    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(tmp_path)
    _install_fmpy_stub(monkeypatch)
    validated: list = []
    monkeypatch.setattr(
        fmpy_ssd, "validate_tree", lambda *args, **kwargs: validated.append(args)
    )

    FMPySimulator().simulate(_fmpy_spec(ssp, validate_ssd=True))

    assert len(validated) >= 1


@pytest.mark.level_contract
def test_fmpy_duplicate_canonical_columns_raise():
    """Two engine columns that collapse to one canonical name must fail."""
    raw = np.zeros(
        2,
        dtype=[
            ("time", np.float64),
            ("root.RampSource.y", np.float64),
            ("RampSource.y", np.float64),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate columns after canonicalization"):
        _canonicalize(raw, None)


# --- FMPy boundary connection resolver (ported from ssp_references) ---------


class MockConnector:
    def __init__(self, name: str, kind: str, path: str) -> None:
        self.name = name
        self.kind = kind
        self._path = path


class MockConnection:
    def __init__(
        self,
        startElement: str | None,
        startConnector: str,
        endElement: str | None,
        endConnector: str,
    ) -> None:
        self.startElement = startElement
        self.startConnector = startConnector
        self.endElement = endElement
        self.endConnector = endConnector


class MockSystem:
    def __init__(self, name: str, connectors: list, connections: list, elements=None) -> None:
        self.name = name
        self.connectors = connectors
        self.connections = connections
        self.elements = elements or []
        self._path = name


@pytest.mark.level_contract
def test_get_connections_empty_system():
    root = MockSystem("system", connectors=[], connections=[], elements=[])
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("fmpy.ssp.ssd.build_path", lambda obj: obj._path)
        monkeypatch.setattr("fmpy.ssp.ssd.find_connectors", lambda system: [])
        assert _get_connections(root) == []


@pytest.mark.level_contract
def test_get_connections_orphan_output_raises():
    out_con = MockConnector("out", "output", "system.out")
    root = MockSystem("system", connectors=[out_con], connections=[], elements=[])
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("fmpy.ssp.ssd.build_path", lambda obj: obj._path)
        monkeypatch.setattr(
            "fmpy.ssp.ssd.find_connectors", lambda system: [out_con]
        )
        with pytest.raises(KeyError, match="Missing connection"):
            _get_connections(root)


@pytest.mark.level_contract
def test_get_connections_root_to_child():
    out_con = MockConnector("out", "output", "system.out")
    in_con = MockConnector("in", "input", "system.child.in")
    child = MockSystem("child", connectors=[in_con], connections=[], elements=[])
    child._path = "system.child"
    connection = MockConnection(
        startElement=None, startConnector="out", endElement="child", endConnector="in"
    )
    root = MockSystem(
        "system", connectors=[out_con], connections=[connection], elements=[child]
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("fmpy.ssp.ssd.build_path", lambda obj: obj._path)
        monkeypatch.setattr(
            "fmpy.ssp.ssd.find_connectors", lambda system: [out_con, in_con]
        )
        result = _get_connections(root)

    assert len(result) == 2
    for start_con, end_con in result:
        assert start_con.name == "out"
        assert end_con.name == "in"


# --- cosim adapter (no binary needed) ---------------------------------------


@pytest.mark.level_contract
class TestCosimAdapter:
    """Unit tests over hand-written cosim CSVs and a fake executable."""

    def test_component_name_strips_timestamp_suffix(self, tmp_path: Path):
        assert (
            component_name_from_csv(tmp_path / "RampSource_20240101_120000_1.csv")
            == "RampSource"
        )
        assert component_name_from_csv(tmp_path / "RampSource.csv") == "RampSource"

    def test_resolve_rejects_non_executable_override(self, tmp_path: Path, monkeypatch):
        binary = tmp_path / "cosim"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o644)
        monkeypatch.setenv("COSIM_BINARY", str(binary))

        assert resolve_cosim_binary() is None

    def test_resolve_accepts_executable_override(self, tmp_path: Path, monkeypatch):
        binary = tmp_path / "cosim"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setenv("COSIM_BINARY", str(binary))

        assert resolve_cosim_binary() == str(binary)

    def test_resolve_rejects_missing_override_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("COSIM_BINARY", str(tmp_path / "nowhere"))

        assert resolve_cosim_binary() is None

    def test_merge_prefixes_components_and_skips_meta_columns(self, tmp_path: Path):
        (tmp_path / "RampSource_20240101_120000_1.csv").write_text(
            "time,stepcount,y\n0,0,0\n1,1,1\n"
        )
        (tmp_path / "Integrator.csv").write_text("time,u,y\n0,0,0\n0.5,1,0.5\n1,1,1\n")

        merged = merge_cosim_csvs(tmp_path)

        assert merged.dtype.names == (
            "time",
            "Integrator.u",
            "Integrator.y",
            "RampSource.y",
        )
        np.testing.assert_allclose(merged["time"], [0.0, 0.5, 1.0])
        # RampSource.y only has samples at 0 and 1; it is interpolated.
        np.testing.assert_allclose(merged["RampSource.y"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(merged["Integrator.y"], [0.0, 0.5, 1.0])

    def test_merge_skips_non_numeric_columns(self, tmp_path: Path):
        (tmp_path / "Ball.csv").write_text("time,y,label\n0,1,a\n1,2,b\n")

        merged = merge_cosim_csvs(tmp_path)

        assert merged.dtype.names == ("time", "Ball.y")

    def test_merge_without_time_column_raises(self, tmp_path: Path):
        (tmp_path / "Ball.csv").write_text("y\n1\n2\n")

        with pytest.raises(ValueError, match="No time column"):
            merge_cosim_csvs(tmp_path)

    def test_merge_without_csvs_raises(self, tmp_path: Path):
        output = tmp_path / "empty"
        output.mkdir()

        with pytest.raises(ValueError, match="No cosim CSV data"):
            merge_cosim_csvs(output)

    def test_hung_binary_times_out(self, tmp_path: Path, monkeypatch):
        binary = tmp_path / "fake_cosim"
        binary.write_text("#!/bin/sh\nsleep 30\n")
        binary.chmod(0o755)
        monkeypatch.setenv("COSIM_BINARY", str(binary))
        ssp = _synthetic_ssp_dir(tmp_path)

        with pytest.raises(TimeoutError, match="did not finish within 0.2 s"):
            CosimSimulator(timeout=0.2).simulate(
                SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0, step_size=0.1)
            )


@pytest.mark.level_contract
def test_backend_timeouts_default_to_600_seconds():
    """A timeout parameter exists without changing normal-run behaviour."""
    assert SSP4SimSimulator().timeout == 600.0
    assert CosimSimulator().timeout == 600.0
