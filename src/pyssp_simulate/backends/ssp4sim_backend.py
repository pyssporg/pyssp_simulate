"""ssp4sim backend: the SSP/FMI engine from the ssp4sim project.

ssp4sim is a C++ engine with Python bindings (``pyssp4sim``) published as
Linux x86_64 wheels on the project's GitHub releases -- it is not on
PyPI, so it cannot be part of this package's install metadata. The
known-good 0.2.11 wheel can be installed directly:

    pip install https://github.com/jkCXf9X4/ssp4sim/releases/download/v0.2.11/pyssp4sim-0.2.11-cp39-abi3-linux_x86_64.whl

The engine is driven by a JSON configuration file and is executed in a
child process for two reasons: it does not reliably return control to
the interpreter after a simulation, and the wheels statically link
libstdc++ while leaving its symbols interposable -- loading
``pyssp4sim`` after a dynamic ``libstdc++.so.6`` is already in the
process (for example via numpy -> OpenBLAS -> libgfortran) cross-wires
the two copies and segfaults during configuration parsing. The child is
therefore run as a bare script that imports ``pyssp4sim`` before
anything else can load a dynamic libstdc++. Additionally, the v0.3.1
wheels contain AVX-512 instructions and crash with SIGILL on CPUs
without AVX-512 support.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from pyssp_simulate.interface import Simulator, import_module_available
from pyssp_simulate.result import SimulationResult, read_result_csv
from pyssp_simulate.runtime import prepare_runtime_ssp
from pyssp_simulate.spec import SimulationSpec

_CHILD_SCRIPT = Path(__file__).resolve().parent / "ssp4sim_child.py"


class SSP4SimSimulator(Simulator):
    name = "ssp4sim"
    install_hint = (
        "Install the pyssp4sim wheel from the ssp4sim GitHub releases "
        "(Linux x86_64 only); see "
        "https://github.com/jkCXf9X4/ssp4sim/releases."
    )

    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        if import_module_available("pyssp4sim"):
            return True, None
        return False, "the `pyssp4sim` package is not installed"

    def simulate(self, spec: SimulationSpec) -> SimulationResult:
        step = spec.effective_step_size()

        with spec.workspace() as work:
            with prepare_runtime_ssp(spec.ssp) as runtime:
                result_path = work / "result.csv"
                config_path = work / "config.json"
                log_path = work / "ssp4sim.log"
                config = _config_payload(
                    spec=spec,
                    runtime_root=runtime.root,
                    ssd_name=runtime.ssd_name,
                    work_dir=work,
                    result_file=result_path,
                    log_file=log_path,
                    step=step,
                )
                config_path.write_text(json.dumps(config, indent=2))

                start = time.perf_counter()
                # Run as a bare script (not ``python -m``): the child must
                # import pyssp4sim before anything loads a dynamic libstdc++
                # (see the module docstring).
                process = subprocess.run(
                    [sys.executable, str(_CHILD_SCRIPT), str(config_path)],
                    capture_output=True,
                    text=True,
                )
                runtime_s = time.perf_counter() - start

                if process.returncode != 0:
                    raise RuntimeError(
                        f"ssp4sim failed with exit code {process.returncode}:\n"
                        f"{process.stderr.strip()}"
                    )
                if not result_path.is_file():
                    raise RuntimeError(
                        f"ssp4sim did not produce a result file at {result_path}"
                    )
                data = read_result_csv(result_path, root_system=runtime.container_names)

        result = SimulationResult(
            data=data,
            backend=self.name,
            spec=spec,
            runtime_s=runtime_s,
            artifacts=(config_path, log_path) if log_path.is_file() else (config_path,),
        )
        if spec.result_file is not None:
            result.write_csv(spec.result_file)
        return result


def _config_payload(
    *,
    spec: SimulationSpec,
    runtime_root: Path,
    ssd_name: str,
    work_dir: Path,
    result_file: Path,
    log_file: Path,
    step: float,
) -> dict:
    """Build the ssp4sim simulation configuration (its own JSON schema)."""
    return {
        "simulation": {
            "ssp": str(runtime_root),
            "ssd": ssd_name,
            "start_time": spec.start_time,
            "stop_time": spec.stop_time,
            "timestep": step,
            "tolerance": spec.tolerance if spec.tolerance is not None else 1e-4,
            "realtime": False,
            "working_dir": str(work_dir),
            "executor": {
                "method": "jacobi",
                "thread_pool_workers": 5,
                "forward_derivatives": True,
                "sub_step": step,
                "jacobi": {"parallel": True, "method": 1},
                "seidel": {"parallel": False},
            },
            "recording": {
                "enable": True,
                "wait_for": True,
                "interval": step,
                "result_file": str(result_file),
                "csv": {
                    "enable": True,
                    "file": str(result_file),
                    "interval": step,
                },
                "influx": {"enable": False},
            },
            "log": {
                "file": str(log_file),
                "level_terminal": "error",
                "level_file": "info",
                "level_json": "disable",
                "level_cutelog": "disable",
                "fmu": False,
            },
        }
    }


SIMULATOR = SSP4SimSimulator
