"""FMPy backend: a pure-Python co-simulation master for SSP packages.

FMPy ships on PyPI for Windows, Linux and macOS and is part of the base
install of ``pyssp_simulate``, which makes this the default backend.

Note that FMPy's SSP support is intentionally simple: it records the
values of the connectors declared in the SSD, simulates FMUs with a
fixed-step Jacobi scheme, and validates SSD files against its bundled
SSP 1.0 schema. Use the other backends for advanced SSP features.
"""

from __future__ import annotations

import time

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
            start = time.perf_counter()
            raw = simulate_ssp(
                str(runtime.archive),
                start_time=spec.start_time,
                stop_time=spec.stop_time,
                step_size=spec.step_size,
            )
            runtime_s = time.perf_counter() - start
            data = _canonicalize(raw, runtime.root_system)

        result = SimulationResult(
            data=data,
            backend=self.name,
            spec=spec,
            runtime_s=runtime_s,
        )
        if spec.result_file is not None:
            result.write_csv(spec.result_file)
        return result


def _canonicalize(raw: np.ndarray, root_system: str | None) -> np.ndarray:
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
