"""FMPy backend: a pure-Python co-simulation master for SSP packages.

FMPy ships on PyPI for Windows, Linux and macOS and is part of the base
install of ``pyssp_simulate``, which makes this the default backend.

Note that FMPy's SSP support is intentionally simple: it records the
values of the connectors declared in the SSD, simulates FMUs with a
fixed-step Jacobi scheme, and validates SSD files against its bundled
SSP 1.0 schema. Use the other backends for advanced SSP features.

The backend always resolves system-boundary connections with its own
traversal (FMPy's resolver trips over connections that pass through the
root system) and applies the parameter set declared in the system
definition. ``SimulationSpec.validate_ssd=False`` skips the bundled
schema validation for the runtime call.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pyssp_simulate.interface import Simulator, import_module_available
from pyssp_simulate.result import SimulationResult, canonicalize_signal
from pyssp_simulate.runtime import prepare_runtime_ssp
from pyssp_simulate.spec import SimulationSpec


class FMPySimulator(Simulator):
    name = "fmpy"
    install_hint = "FMPy is a core dependency: `pip install fmpy`."

    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        if import_module_available("fmpy"):
            return True, None
        return False, "the `fmpy` package is not installed"

    def simulate(self, spec: SimulationSpec) -> SimulationResult:
        from fmpy.ssp.simulation import simulate_ssp

        with prepare_runtime_ssp(spec.ssp) as runtime:
            with _patched_ssp_loader(validate=spec.validate_ssd):
                parameter_set = _parameter_set_from_ssd(
                    runtime.archive, validate=spec.validate_ssd
                )
                start = time.perf_counter()
                raw = simulate_ssp(
                    str(runtime.archive),
                    start_time=spec.start_time,
                    stop_time=spec.stop_time,
                    step_size=spec.step_size,
                    parameter_set=parameter_set,
                )
                runtime_s = time.perf_counter() - start
            data = _canonicalize(raw, runtime.container_names)

        result = SimulationResult(
            data=data,
            backend=self.name,
            spec=spec,
            runtime_s=runtime_s,
        )
        if spec.result_file is not None:
            result.write_csv(spec.result_file)
        return result


def _parameter_set_from_ssd(archive: Path, *, validate: bool) -> object | None:
    """Return the parameter set declared by the system definition.

    Applies only the parameter set declared in the system definition (the
    SSD parameter binding, with its SSM mapping applied to target names).
    If the SSD declares no parameter binding, no parameter set is passed:
    nothing is invented, no external SSV the system definition does not
    reference is merged, and none is applied.

    :raises ValueError: when an SSM mapping source is absent from the
        declared parameter set.
    """
    from fmpy.ssp.ssd import read_ssd

    ssd = read_ssd(str(archive), validate=validate)
    binding = next(
        (candidate for candidate in ssd.system.parameterBindings if candidate.parameterValues),
        None,
    )
    if binding is None:
        return None
    return _apply_fmpy_mapping(binding)


def _apply_fmpy_mapping(binding) -> object | None:
    """Resolve FMPy's externally loaded SSM into target-named parameters."""
    from fmpy.ssp.ssd import Parameter, ParameterSet

    if not binding.parameterValues:
        return None
    parameter_set = binding.parameterValues[0]
    mappings = [
        entry for mapping in binding.parameterMapping for entry in mapping.mappingEntries
    ]
    if not mappings:
        return parameter_set

    by_name = {parameter.name: parameter for parameter in parameter_set.parameters}
    resolved = ParameterSet(name=parameter_set.name)
    for entry in mappings:
        source = by_name.get(entry.source)
        if source is None:
            raise ValueError(
                f"FMPy mapping source '{entry.source}' is not present in "
                f"parameter set '{parameter_set.name}'"
            )
        resolved.parameters.append(
            Parameter(
                name=entry.target,
                type=source.type,
                value=source.value,
                unit=source.unit,
            )
        )
    return resolved


@contextmanager
def _patched_ssp_loader(*, validate: bool) -> Iterator[None]:
    """Install the boundary-safe connection resolver for the runtime call.

    When ``validate`` is False, FMPy's bundled SSP/SSV/SSM schema validation
    is neutralised for the duration of the call as well.
    """
    from fmpy.ssp import simulation as fmpy_simulation
    from fmpy.ssp import ssd as fmpy_ssd

    original_validate_tree = fmpy_ssd.validate_tree
    original_get_connections = fmpy_simulation.get_connections

    def _no_validate(*args, **kwargs):
        return None

    if not validate:
        fmpy_ssd.validate_tree = _no_validate
    fmpy_simulation.get_connections = _get_connections
    try:
        yield
    finally:
        fmpy_ssd.validate_tree = original_validate_tree
        fmpy_simulation.get_connections = original_get_connections


def _get_connections(system, connectors=None):
    """Resolve FMPy SSP connections, including system-boundary ones.

    FMPy's built-in resolver returns the wrong pairs when a connection passes
    through the root system (for example ``input -> root output -> child``).
    This mirrors its traversal and fixes that branch, so the runtime archive
    can be instantiated without changing the model.
    """
    from fmpy.ssp.ssd import System, build_path, find_connectors

    if connectors is None:
        connectors = {}
        for connector in find_connectors(system):
            connectors[build_path(connector)] = connector

    cons = []

    for connector in system.connectors:
        if connector.kind == "output":
            start_p = None
            end_p = None
            for connection in system.connections:
                if connection.endElement is None and connection.endConnector == connector.name:
                    start_p = (
                        build_path(system)
                        + "."
                        + connection.startElement
                        + "."
                        + connection.startConnector
                    )
                    end_p = build_path(connector)
                    break
                if connection.startElement is None and connection.startConnector == connector.name:
                    start_p = build_path(connector)
                    end_p = (
                        build_path(system)
                        + "."
                        + connection.endElement
                        + "."
                        + connection.endConnector
                    )
                    break
            if start_p is None or end_p is None:
                raise KeyError(f"Missing connection for connector {build_path(connector)}")
            cons.append((connectors[start_p], connectors[end_p]))

    for element in system.elements:
        for connector in element.connectors:
            if connector.kind == "input":
                end_p = build_path(element) + "." + connector.name
                start_p = None
                for connection in system.connections:
                    if (
                        connection.endElement == element.name
                        and connection.endConnector == connector.name
                    ):
                        start_p = build_path(system)
                        if connection.startElement is not None:
                            start_p += "." + connection.startElement
                        start_p += "." + connection.startConnector
                        break
                    if (
                        connection.startElement == element.name
                        and connection.startConnector == connector.name
                    ):
                        start_p = build_path(system)
                        if connection.endElement is not None:
                            start_p += "." + connection.endElement
                        start_p += "." + connection.endConnector
                        break
                if start_p is None:
                    path = build_path(element) + "." + connector.name
                    raise KeyError(f"Missing connection for connector {path}")
                cons.append((connectors[start_p], connectors[end_p]))

    for element in system.elements:
        if isinstance(element, System):
            cons += _get_connections(element, connectors=connectors)

    return cons


def _canonicalize(raw: np.ndarray, root_system: str | Sequence[str] | None) -> np.ndarray:
    """Rename FMPy's result fields to canonical signal names."""
    names = list(raw.dtype.names or ())
    canonical = {name: canonicalize_signal(name, root_system) for name in names}
    if all(new == old for old, new in canonical.items()):
        return raw

    dtype = [(canonical[name], raw.dtype[name]) for name in names]
    data = np.empty_like(raw, dtype=dtype)
    for old, new in canonical.items():
        data[new] = raw[old]
    return data


SIMULATOR = FMPySimulator
