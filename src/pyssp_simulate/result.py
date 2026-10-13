"""Normalized simulation results and CSV I/O.

Every backend returns a :class:`SimulationResult` holding a structured
NumPy array. The array's field names are *canonical* signal names of the
form ``<component>.<signal>`` (``"time"`` for the independent variable),
regardless of how each engine names its result columns.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyssp_simulate.spec import SimulationSpec, spec_summary

TIME_SIGNAL = "time"

# Leading path segments engines prepend to signal names that carry no
# meaning for users: OMSimulator's model and root-system names ("model",
# "root"), and the generic "fmu" prefix some tools use.
_DROP_PREFIXES = ("root", "fmu", "model")


def canonicalize_signal(name: str, root_system: str | None = None) -> str:
    """Normalize an engine result column name to ``<component>.<signal>``.

    Strips surrounding whitespace and quotes and removes leading path
    segments that refer to the root system rather than a component, while
    always keeping at least two segments for qualified names.
    """
    normalized = name.strip().strip('"').strip("'")
    if normalized.lower() == TIME_SIGNAL:
        return TIME_SIGNAL

    parts = [part for part in normalized.split(".") if part]
    if len(parts) < 2:
        return normalized

    droppable = set(_DROP_PREFIXES)
    if root_system:
        droppable.add(root_system)

    while len(parts) > 2 and parts[0] in droppable:
        parts.pop(0)

    return ".".join(parts)


def array_from_columns(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Build a structured array from an ordered mapping of signal columns."""
    if not columns:
        raise ValueError("Cannot build a result array from zero columns")

    names = list(columns)
    if TIME_SIGNAL in columns:
        length = len(columns[TIME_SIGNAL])
    else:
        length = len(next(iter(columns.values())))
    dtype = [(name, np.float64) for name in names]
    array = np.empty(length, dtype=dtype)
    for name, values in columns.items():
        array[name] = values
    return array


def read_result_csv(path: Path | str, root_system: str | None = None) -> np.ndarray:
    """Read a numeric result CSV into a structured array.

    The first row must be the header; all values are parsed as floats.
    Header names are canonicalized with :func:`canonicalize_signal`.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Result CSV {path} is empty") from error
        rows = [row for row in reader if row]

    header = [canonicalize_signal(name, root_system) for name in raw_header]
    if len(set(header)) != len(header):
        raise ValueError(f"Duplicate columns after canonicalization in {path}: {header}")

    if not rows:
        return array_from_columns({name: np.array([], dtype=np.float64) for name in header})

    values = np.array([[float(cell) for cell in row] for row in rows], dtype=np.float64)
    if values.shape[1] != len(header):
        raise ValueError(f"Ragged rows in result CSV {path}")
    return array_from_columns({name: values[:, index] for index, name in enumerate(header)})


def write_result_csv(path: Path | str, data: np.ndarray) -> Path:
    """Write a structured array as a plain numeric CSV with a header row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(data.dtype.names or ())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row in data:
            writer.writerow([float(row[name]) for name in names])
    return path


@dataclass
class SimulationResult:
    """The outcome of one simulation run.

    :param data: structured array; ``"time"`` plus one field per signal.
    :param backend: name of the backend that produced the result.
    :param spec: the simulation request this result answers, if known.
    :param runtime_s: wall-clock seconds spent simulating, if known.
    :param artifacts: files produced by the run (config, logs, raw engine
        output) that were intentionally kept.
    """

    data: np.ndarray
    backend: str
    spec: SimulationSpec | None = None
    runtime_s: float | None = None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def signals(self) -> tuple[str, ...]:
        """All signal names, ``"time"`` excluded."""
        names = list(self.data.dtype.names or ())
        return tuple(name for name in names if name != TIME_SIGNAL)

    @property
    def time(self) -> np.ndarray:
        return self.data[TIME_SIGNAL]

    def __getitem__(self, signal: str) -> np.ndarray:
        if signal not in (self.data.dtype.names or ()):
            raise KeyError(f"Signal {signal!r} not in result; available: {self.signals}")
        return self.data[signal]

    def __len__(self) -> int:
        return len(self.data)

    def write_csv(self, path: Path | str) -> Path:
        """Write the result to a numeric CSV and record it as an artifact."""
        target = write_result_csv(path, self.data)
        if target not in self.artifacts:
            self.artifacts = (*self.artifacts, target)
        return target

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        backend: str = "unknown",
        spec: SimulationSpec | None = None,
        runtime_s: float | None = None,
        artifacts: tuple[Path, ...] = (),
    ) -> SimulationResult:
        return cls(
            data=read_result_csv(path),
            backend=backend,
            spec=spec,
            runtime_s=runtime_s,
            artifacts=artifacts,
        )

    def summary(self) -> dict:
        """A compact, JSON-friendly description of the result."""
        return {
            "backend": self.backend,
            "signals": list(self.signals),
            "samples": len(self.data),
            "time_range": [float(self.time[0]), float(self.time[-1])]
            if len(self.data)
            else None,
            "runtime_s": self.runtime_s,
            "spec": spec_summary(self.spec) if self.spec else None,
        }
