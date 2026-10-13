"""OMSimulator backend: OpenModelica's FMI-based co-simulation environment.

The Python package is published on PyPI as ``OMSimulator`` (version 3.x
provides the ``OMSimulator`` module used here). The package is an sdist
that downloads engine binaries at install time, which works on current
platforms but fails on distributions whose build artifacts are no
longer published -- that is why it is an optional extra rather than a
core dependency. See the README for manual installation instructions.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pyssp_simulate.interface import Simulator, import_module_available
from pyssp_simulate.result import SimulationResult, read_result_csv
from pyssp_simulate.runtime import prepare_runtime_ssp
from pyssp_simulate.spec import SimulationSpec


class OMSimulatorSimulator(Simulator):
    name = "omsimulator"
    install_hint = (
        "Install with `pip install pyssp_simulate[omsimulator]`, or install "
        "OMSimulator manually (see https://github.com/pyssporg/pyssp_simulate#backends)."
    )

    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        if import_module_available("OMSimulator"):
            return True, None
        return False, "the `OMSimulator` package is not installed"

    def simulate(self, spec: SimulationSpec) -> SimulationResult:
        from OMSimulator import SSP as OMSSP
        from OMSimulator import Settings

        Settings.suppressPath = True

        with spec.workspace() as work:
            with prepare_runtime_ssp(spec.ssp) as runtime:
                result_path = work / "result.csv"
                # OMSimulator writes logging artifacts relative to the
                # current directory, so run it inside the workspace.
                with _temporary_cwd(work):
                    model = OMSSP(str(runtime.archive))
                    instance = model.instantiate()
                    try:
                        instance.setStartTime(spec.start_time)
                        instance.setStopTime(spec.stop_time)
                        instance.setFixedStepSize(spec.effective_step_size())
                        instance.setLoggingInterval(spec.effective_step_size())
                        if spec.tolerance is not None:
                            instance.setTolerance(spec.tolerance)
                        instance.setResultFile(str(result_path))
                        start = time.perf_counter()
                        instance.initialize()
                        instance.simulate()
                        instance.terminate()
                        runtime_s = time.perf_counter() - start
                    finally:
                        instance.delete()

                if not result_path.is_file():
                    raise RuntimeError(
                        f"OMSimulator did not produce a result file at {result_path}"
                    )
                data = read_result_csv(result_path, root_system=runtime.root_system)

        result = SimulationResult(
            data=data,
            backend=self.name,
            spec=spec,
            runtime_s=runtime_s,
        )
        if spec.result_file is not None:
            result.write_csv(spec.result_file)
        return result


@contextmanager
def _temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


SIMULATOR = OMSimulatorSimulator
