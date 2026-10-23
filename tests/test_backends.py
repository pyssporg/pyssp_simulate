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
    START_TIME,
    STEP_SIZE,
    STOP_TIME,
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
from pyssp_simulate.backends.fmpy_backend import _get_connections
from pyssp_simulate.backends.omsimulator_backend import OMSimulatorSimulator
from pyssp_simulate.backends.ssp4sim_backend import (
    DIAGNOSTIC_ARTIFACTS,
    SSP4SimSimulator,
)
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

    def fake_run(command, capture_output=True, text=True):
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


def test_ssp4sim_requires_an_explicit_tolerance(tmp_path: Path):
    """No silent default: the engine requires the key, so the caller must pass it."""
    ssp = _synthetic_ssp_dir(tmp_path)

    with pytest.raises(ValueError, match="requires an explicit tolerance"):
        SSP4SimSimulator().simulate(
            SimulationSpec(ssp=ssp, start_time=0.0, stop_time=1.0, step_size=0.1)
        )


class _FakeOMInstance:
    def __init__(self, calls: dict):
        self.calls = calls
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
        self.result_path.write_text("time,Source.y\n0.0,0.0\n1.0,1.0\n")

    def terminate(self):
        pass

    def delete(self):
        pass


def _install_omsimulator_stub(monkeypatch: pytest.MonkeyPatch) -> dict:
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
            return _FakeOMInstance(calls)

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


def test_omsimulator_fixed_step_true_sets_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ssp = _synthetic_ssp_dir(tmp_path)
    calls = _install_omsimulator_stub(monkeypatch)

    OMSimulatorSimulator().simulate(_omsimulator_spec(ssp))

    assert calls["logging"] == [pytest.approx(0.1)]
    assert calls["fixed"] == [pytest.approx(0.1)]


def test_omsimulator_options_reach_the_engine_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ssp = _synthetic_ssp_dir(tmp_path)
    calls = _install_omsimulator_stub(monkeypatch)
    options = ("--wallTime=true", "--emitEvents=false")

    OMSimulatorSimulator().simulate(_omsimulator_spec(ssp, omsimulator_options=options))

    assert calls["options"] == list(options)


def test_omsimulator_rejected_option_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ssp = _synthetic_ssp_dir(tmp_path)
    _install_omsimulator_stub(monkeypatch)

    with pytest.raises(RuntimeError, match="command line option"):
        OMSimulatorSimulator().simulate(
            _omsimulator_spec(ssp, omsimulator_options=("--bad=true",))
        )


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


def test_fmpy_passes_the_declared_parameter_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(tmp_path, binding=_inline_binding({"height": 2.0}))
    observed = _install_fmpy_stub(monkeypatch)

    FMPySimulator().simulate(_fmpy_spec(ssp))

    assert _parameter_names(observed["parameter_set"]) == ["height"]


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


def test_fmpy_without_parameter_binding_passes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pyssp_simulate.backends.fmpy_backend import FMPySimulator

    ssp = _synthetic_ssp_dir(tmp_path)
    observed = _install_fmpy_stub(monkeypatch)

    FMPySimulator().simulate(_fmpy_spec(ssp))

    assert observed["parameter_set"] is None


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


def test_get_connections_empty_system():
    root = MockSystem("system", connectors=[], connections=[], elements=[])
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("fmpy.ssp.ssd.build_path", lambda obj: obj._path)
        monkeypatch.setattr("fmpy.ssp.ssd.find_connectors", lambda system: [])
        assert _get_connections(root) == []


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

