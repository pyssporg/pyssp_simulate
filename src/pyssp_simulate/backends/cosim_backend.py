"""cosim backend: the Open Simulation Platform command-line co-simulator.

**Status: experimental.** Validated against cosim v0.8.0 on Linux x86_64;
the v0.9.0 release binaries require GLIBC >= 2.38, and other platforms
and versions are untested.

`cosim <https://github.com/open-simulation-platform/cosim-cli>`_ is a
C++ tool built on libcosim that runs system structures given in OSP or
SSP format. It is not pip-installable; download a release binary from
the project's GitHub page and make sure it is on ``PATH``, or point the
``COSIM_BINARY`` environment variable at it (when set, ``COSIM_BINARY``
is used exclusively -- a stale value disables the backend rather than
falling back to ``PATH``).

Notes:

* cosim has no command-line step-size option: libcosim reads the master
  step from an OSP annotation in the SSD (``osp:FixedStepAlgorithm`` /
  ``baseStepSize``), so ``SimulationSpec.step_size`` does not apply to
  this backend. Packages without such an annotation fail in cosim itself
  (v0.8.0 exits with a segfault).
* cosim writes one CSV file per component, named
  ``<component>_<timestamp>.csv`` by default, plus ``_metadata.yaml``
  files; the backend merges the CSVs into a single result and prefixes
  signals with the component name.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from pyssp_simulate.interface import Simulator
from pyssp_simulate.result import (
    TIME_SIGNAL,
    SimulationResult,
    array_from_columns,
    canonicalize_signal,
)
from pyssp_simulate.runtime import prepare_runtime_ssp
from pyssp_simulate.spec import SimulationSpec

# Columns cosim writes that are not model signals.
_COSIM_META_COLUMNS = {"stepcount"}


class CosimSimulator(Simulator):
    name = "cosim"
    install_hint = (
        "Install the cosim binary from "
        "https://github.com/open-simulation-platform/cosim-cli/releases "
        "and put it on PATH (or set COSIM_BINARY)."
    )

    #: Seconds to wait for the engine child before giving up. A hung engine
    #: must fail the run instead of blocking the caller forever.
    DEFAULT_TIMEOUT_S = 600.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout = timeout

    @classmethod
    def available(cls) -> tuple[bool, str | None]:
        if resolve_cosim_binary() is not None:
            return True, None
        return False, "the `cosim` executable was not found on PATH or via COSIM_BINARY"

    def simulate(self, spec: SimulationSpec) -> SimulationResult:
        """Run one simulation with the cosim binary.

        :raises RuntimeError: when the binary is missing, exits non-zero, or
            writes no result CSVs.
        :raises TimeoutError: when the child does not finish within
            ``timeout`` seconds (default :attr:`DEFAULT_TIMEOUT_S`).
        """
        binary = resolve_cosim_binary()
        if binary is None:
            raise RuntimeError("cosim binary not found")

        with spec.workspace() as work:
            output_dir = work / "cosim_output"
            with prepare_runtime_ssp(spec.ssp) as runtime:
                command = [
                    str(binary),
                    "run",
                    str(runtime.archive),
                    f"--begin-time={spec.start_time}",
                    f"--end-time={spec.stop_time}",
                    f"--output-dir={output_dir}",
                ]
                start = time.perf_counter()
                try:
                    process = subprocess.run(
                        command, capture_output=True, text=True, timeout=self.timeout
                    )
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError(
                        f"cosim did not finish within {self.timeout} s"
                    ) from error
                runtime_s = time.perf_counter() - start
                if process.returncode != 0:
                    raise RuntimeError(
                        f"cosim failed with exit code {process.returncode}:\n"
                        f"{process.stderr.strip() or process.stdout.strip()}"
                    )

            if not output_dir.is_dir() or not list(output_dir.glob("*.csv")):
                raise RuntimeError(f"cosim did not write result CSVs to {output_dir}")
            data = merge_cosim_csvs(output_dir)

        result = SimulationResult(
            data=data,
            backend=self.name,
            spec=spec,
            runtime_s=runtime_s,
        )
        if spec.result_file is not None:
            result.write_csv(spec.result_file)
        return result


def resolve_cosim_binary() -> str | None:
    """Find the cosim executable: ``COSIM_BINARY`` if set (exclusively --
    no ``PATH`` fallback when it points nowhere), else ``PATH``.

    A ``COSIM_BINARY`` that exists but is not an executable file is
    rejected, so a stale or non-executable override disables the backend
    instead of failing inside :func:`subprocess.run`.
    """
    from_env = os.environ.get("COSIM_BINARY")
    if from_env:
        candidate = Path(from_env)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return from_env
        return None
    return shutil.which("cosim")


_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}_\d{6}_\d+$")


def component_name_from_csv(path: Path) -> str:
    """Recover the component name from a cosim CSV file name.

    cosim names files ``<component>.csv`` or, by default,
    ``<component>_<timestamp>.csv``.
    """
    stem = path.name[: -len(".csv")] if path.name.endswith(".csv") else path.stem
    return _TIMESTAMP_SUFFIX.sub("", stem)


def merge_cosim_csvs(output_dir: Path) -> np.ndarray:
    """Merge cosim's per-component CSVs into one structured array.

    Each signal is resampled (linearly) onto the union of the per-file
    time grids so all signals share one time axis.
    """
    signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grids: list[np.ndarray] = []

    for csv_path in sorted(output_dir.glob("*.csv")):
        component = component_name_from_csv(csv_path)
        with csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = [name.strip() for name in next(reader)]
            except StopIteration:
                continue
            rows = [row for row in reader if row]
        if not rows:
            continue

        time_index = next(
            (i for i, name in enumerate(header) if name.lower() == "time"), None
        )
        if time_index is None:
            raise ValueError(f"No time column in cosim CSV {csv_path}")

        grid = np.array([float(row[time_index]) for row in rows], dtype=np.float64)
        grids.append(grid)

        for index, name in enumerate(header):
            canonical = canonicalize_signal(name)
            if canonical == TIME_SIGNAL or canonical.lower() in _COSIM_META_COLUMNS:
                continue
            values = _tolerant_float_column(rows, index)
            if values is not None:
                signals[f"{component}.{canonical}"] = (grid, values)

    if not grids:
        raise ValueError(f"No cosim CSV data found in {output_dir}")

    union_grid = np.unique(np.concatenate(grids))
    columns: dict[str, np.ndarray] = {TIME_SIGNAL: union_grid}
    for signal, (grid, values) in signals.items():
        if np.array_equal(grid, union_grid):
            columns[signal] = values
        else:
            columns[signal] = np.interp(union_grid, grid, values)
    return array_from_columns(columns)


def _tolerant_float_column(rows: list[list[str]], index: int) -> np.ndarray | None:
    """Parse a column as floats; ``None`` if any value is non-numeric."""
    try:
        return np.array([float(row[index]) for row in rows], dtype=np.float64)
    except (ValueError, IndexError):
        return None


SIMULATOR = CosimSimulator
