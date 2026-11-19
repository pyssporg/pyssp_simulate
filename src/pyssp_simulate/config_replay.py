"""Run the ssp4sim engine against an existing configuration file.

Some callers already own their ssp4sim JSON configuration (paths, executor
settings, recording policy) and need it executed verbatim -- for example a
scenario tool that wrote the file earlier and must not have it rewritten.
The normal :class:`~pyssp_simulate.spec.SimulationSpec` path cannot express
that: it builds a fresh configuration in a scratch directory.

:func:`replay_ssp4sim_config` runs the engine child on the given file without
reading it back into a spec and without touching it. Relative ``ssp``,
``working_dir``, ``recording.csv.file``, and ``log.file`` values keep the
engine's native meaning: they resolve against the process working directory,
exactly as when the engine is invoked directly.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pyssp_simulate.backends.ssp4sim_backend import (
    SSP4SimSimulator,
    _execute_config,
)
from pyssp_simulate.result import SimulationResult, read_result_csv
from pyssp_simulate.runtime import read_container_names_from_bytes

_SSD_DEFAULT = "SystemStructure.ssd"


def replay_ssp4sim_config(
    config_path: Path | str,
    *,
    timeout: float = SSP4SimSimulator.DEFAULT_TIMEOUT_S,
) -> SimulationResult:
    """Simulate using an existing ssp4sim JSON configuration.

    The file is passed to the same engine child as the ``ssp4sim`` backend
    and is never modified. The result CSV named by
    ``simulation.recording.csv.file`` (or the older
    ``simulation.recording.result_file``) is read back; when neither is set,
    ``simulation.working_dir`` (default ``.``) is searched for ``result.csv``.

    :param config_path: ssp4sim configuration file to execute.
    :param timeout: seconds to wait for the engine child.
    :raises FileNotFoundError: if ``config_path`` does not exist.
    :raises ValueError: if the file is not an ssp4sim configuration object.
    :raises TimeoutError: when the child does not finish within ``timeout``.
    :raises RuntimeError: when the child exits non-zero or writes no result
        file.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"ssp4sim config not found: {path}")

    config = json.loads(path.read_text())
    simulation = config.get("simulation") if isinstance(config, dict) else None
    if not isinstance(simulation, dict):
        raise ValueError(f"{path} has no ssp4sim 'simulation' object")

    recording = simulation.get("recording") or {}
    csv_config = recording.get("csv") or {}
    working_dir = Path(simulation.get("working_dir", "."))
    result_path = Path(
        csv_config.get("file")
        or recording.get("result_file")
        or working_dir / "result.csv"
    )
    log_file = (simulation.get("log") or {}).get("file")
    log_path = Path(log_file) if log_file else working_dir / "sim.log"

    runtime_s = _execute_config(path, timeout)

    if not result_path.is_file():
        raise RuntimeError(
            f"ssp4sim did not produce a result file at {result_path}"
        )
    data = read_result_csv(
        result_path, root_system=_container_names(simulation)
    )

    artifacts = (path,)
    if log_path.is_file():
        artifacts = (path, log_path)
    return SimulationResult(
        data=data,
        backend=SSP4SimSimulator.name,
        spec=None,
        runtime_s=runtime_s,
        artifacts=artifacts,
    )


def _container_names(simulation: dict) -> tuple[str, ...]:
    """Read the SSD container names for a configured ``ssp`` package."""
    ssp_value = simulation.get("ssp")
    if not ssp_value:
        return ()
    ssp = Path(ssp_value)
    ssd_name = simulation.get("ssd") or _SSD_DEFAULT

    if ssp.is_dir():
        ssd_path = ssp / ssd_name
        return (
            read_container_names_from_bytes(ssd_path.read_bytes())
            if ssd_path.is_file()
            else ()
        )

    if ssp.is_file():
        try:
            with zipfile.ZipFile(ssp) as archive:
                members = archive.namelist()
                if ssd_name in members:
                    member = ssd_name
                else:
                    candidates = sorted(
                        name
                        for name in members
                        if "/" not in name and name.endswith(".ssd")
                    )
                    if not candidates:
                        return ()
                    member = candidates[0]
                return read_container_names_from_bytes(archive.read(member))
        except (KeyError, OSError, zipfile.BadZipFile):
            return ()
    return ()
