"""Normalized simulation results and CSV I/O.

Every backend returns a :class:`SimulationResult` holding a structured
NumPy array. The array's field names are *canonical* signal names of the
form ``<component>.<signal>`` (``"time"`` for the independent variable),
regardless of how each engine names its result columns.

Field order is canonical and engine-independent: ``"time"`` first when
present, then every other field sorted by code point. Engines may emit
different column sets, but a column they share always lands in the same
relative position, so results stay comparable across engines.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyssp_simulate.spec import SimulationSpec, spec_summary

TIME_SIGNAL = "time"

# Leading path segments engines prepend to signal names that carry no
# meaning for users: OMSimulator's model and root-system names ("model",
# "root"), and the generic "fmu" prefix some tools use.
_DROP_PREFIXES = ("root", "fmu", "model")


def canonicalize_signal(
    name: str, root_system: str | Sequence[str] | None = None
) -> str:
    """Normalize an engine result column name to ``<component>.<signal>``.

    Strips surrounding whitespace and quotes, maps the independent variable
    to ``"time"`` (case-insensitively — engines also emit ``"Time"``), and
    drops leading path segments that refer to containers rather than
    components: the literal segments ``root``, ``fmu``, and ``model``, plus
    the container name(s) given in ``root_system`` — a single name or a
    sequence, for example the SSD and system names OMSimulator prefixes
    its result columns with. At least two segments are always kept for
    qualified names, so a two-segment name like ``root.y`` is returned
    unchanged. Empty segments are dropped.
    """
    normalized = name.strip().strip('"').strip("'")
    if normalized.lower() == TIME_SIGNAL:
        return TIME_SIGNAL

    parts = [part for part in normalized.split(".") if part]
    if len(parts) < 2:
        return normalized

    droppable = set(_DROP_PREFIXES)
    if root_system:
        roots = (root_system,) if isinstance(root_system, str) else root_system
        droppable.update(root for root in roots if root)

    while len(parts) > 2 and parts[0] in droppable:
        parts.pop(0)

    # ``fmu`` is a generic container prefix even when only one signal segment
    # follows it (engines emit both ``fmu.h`` and ``fmu.Component.h``).
    if len(parts) >= 2 and parts[0] == "fmu":
        parts.pop(0)

    return ".".join(parts)


def array_from_columns(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Build a structured array from a mapping of signal columns.

    Field order is canonical and independent of the mapping's insertion
    order: ``"time"`` first when present, then the remaining fields sorted
    by code point.
    """
    if not columns:
        raise ValueError("Cannot build a result array from zero columns")

    names = sorted(columns)
    if TIME_SIGNAL in names:
        names.remove(TIME_SIGNAL)
        names.insert(0, TIME_SIGNAL)
    if TIME_SIGNAL in columns:
        length = len(columns[TIME_SIGNAL])
    else:
        length = len(next(iter(columns.values())))
    dtype = [(name, np.float64) for name in names]
    array = np.empty(length, dtype=dtype)
    for name, values in columns.items():
        array[name] = values
    return array


def read_result_csv(
    path: Path | str,
    root_system: str | Sequence[str] | None = None,
    *,
    prefix: str | None = None,
) -> np.ndarray:
    """Read a numeric result CSV into a structured array.

    The first row must be the header; all values are parsed as floats.
    Header names are canonicalized with :func:`canonicalize_signal`, and
    fields come back in canonical order (``"time"`` first, then sorted).

    :param path: CSV file to read.
    :param root_system: container level name(s) to strip from header names,
        for engine output that prefixes signals with the SSD name and/or
        system names (OMSimulator does). Accepts one name or a sequence.
    :param prefix: exact leading text to strip before canonicalization, for
        reference CSVs such as ``default.BouncingBall.h``.
    :raises ValueError: if the file is empty, has ragged rows, or has
        duplicate columns after canonicalization.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Result CSV {path} is empty") from error
        rows = [row for row in reader if row]

    header = []
    column_indices = []
    for index, name in enumerate(raw_header):
        stripped = name.strip()
        if prefix and stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
        canonical = canonicalize_signal(stripped, root_system)
        if canonical in header:
            if canonical == TIME_SIGNAL or prefix:
                # Engines emit both ``time`` and ``<prefix>.time``; reference
                # CSVs do the same with ``default.<model>.time``.
                continue
            raise ValueError(
                f"Duplicate columns after canonicalization in {path}: {header + [canonical]}"
            )
        header.append(canonical)
        column_indices.append(index)

    if not rows:
        return array_from_columns({name: np.array([], dtype=np.float64) for name in header})

    values = np.array([[float(cell) for cell in row] for row in rows], dtype=np.float64)
    if values.shape[1] != len(raw_header):
        raise ValueError(f"Ragged rows in result CSV {path}")
    return array_from_columns(
        {
            name: values[:, source_index]
            for name, source_index in zip(header, column_indices)
        }
    )


def read_dymola_mat(path: Path | str) -> np.ndarray:
    """Read a Dymola ``dsres`` MAT file into a structured result array.

    Decodes ``Aclass``/``name``/``description``/``dataInfo``/``data_1``/
    ``data_2``; the signed row index in ``dataInfo`` encodes negation; a
    ``data_1`` row is a constant series, a ``data_2`` row is a trajectory.
    Returns ``time`` plus one field per named variable, ready for
    :func:`write_result_csv`. Requires the ``dymola`` extra (scipy).

    Variables whose ``dataInfo`` source matrix is neither ``1`` nor ``2``
    are skipped, and a variable named ``time`` is not duplicated (the time
    axis always comes from the first row of ``data_2``).

    :param path: MAT file written by Dymola.
    :raises FileNotFoundError: if ``path`` does not exist.
    :raises KeyError: if a required MAT matrix is missing.
    """
    from scipy.io import loadmat

    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    data = loadmat(mat_path, chars_as_strings=False)
    required = {"Aclass", "name", "description", "dataInfo", "data_1", "data_2"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Missing expected MAT keys: {', '.join(sorted(missing))}")

    # The character matrices are decoded for validation; the result array
    # carries only the names and their values.
    _decode_char_matrix(data["Aclass"])
    names = _decode_char_matrix(data["name"])
    _decode_char_matrix(data["description"])

    data_info = np.asarray(data["dataInfo"], dtype=np.int64)
    data_1 = np.asarray(data["data_1"], dtype=np.float64)
    data_2 = np.asarray(data["data_2"], dtype=np.float64)
    if data_2.shape[0] < 1:
        raise ValueError("data_2 does not contain a time row")

    time = np.asarray(data_2[0, :], dtype=np.float64)
    columns: dict[str, np.ndarray] = {TIME_SIGNAL: time}

    for index, name in enumerate(names):
        if name == TIME_SIGNAL:
            continue

        source_matrix = int(data_info[0, index])
        signed_source_row = int(data_info[1, index])
        # Dymola-style MAT aliases encode negation in the signed row index.
        source_row = abs(signed_source_row)
        sign = -1.0 if signed_source_row < 0 else 1.0

        if source_matrix == 1:
            if source_row < 1 or source_row > data_1.shape[0]:
                raise IndexError(f"Invalid data_1 row {source_row} for variable {name}")
            value = sign * float(data_1[source_row - 1, 0])
            columns[name] = np.full(time.shape, value, dtype=np.float64)
        elif source_matrix == 2:
            if source_row < 1 or source_row > data_2.shape[0]:
                raise IndexError(f"Invalid data_2 row {source_row} for variable {name}")
            columns[name] = sign * np.asarray(data_2[source_row - 1, :], dtype=np.float64)

    return array_from_columns(columns)


def _decode_char_matrix(value: np.ndarray) -> list[str]:
    """Decode a MATLAB/Dymola character matrix into one string per column."""
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D character matrix, got shape {value.shape}")
    strings = []
    for index in range(value.shape[1]):
        codes = (
            chr(int(code)) if not isinstance(code, str) else str(code)
            for code in value[:, index]
        )
        strings.append("".join(codes).rstrip(" \x00"))
    return strings


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
    :param artifacts: files intentionally kept from the run; contents are
        backend-dependent (see :attr:`artifacts`).
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
        """The independent variable as a NumPy array."""
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
        *,
        strip_prefix: str | None = None,
    ) -> SimulationResult:
        """Read a result previously written by :meth:`write_csv`.

        Header names are canonicalized; the standard ``model``/``root``/
        ``fmu`` prefixes are dropped when a component segment remains. Use
        ``strip_prefix`` to remove an explicit leading prefix such as
        ``"fmu."`` from two-segment names.
        """
        return cls(
            data=read_result_csv(path, prefix=strip_prefix),
            backend=backend,
            spec=spec,
            runtime_s=runtime_s,
            artifacts=artifacts,
        )

    @classmethod
    def from_reference_csv(
        cls,
        path: Path | str,
        *,
        model_name: str | None = None,
        prefix: str | None = None,
        backend: str = "reference",
        spec: SimulationSpec | None = None,
        runtime_s: float | None = None,
        artifacts: tuple[Path, ...] = (),
    ) -> SimulationResult:
        """Read a trusted reference CSV, stripping its model prefix.

        Reference trajectories commonly name columns
        ``default.<model>.<signal>``. Pass ``model_name`` for that convention,
        or an explicit ``prefix`` for another one.
        """
        if prefix is None and model_name:
            prefix = f"default.{model_name}."
        return cls(
            data=read_result_csv(path, prefix=prefix),
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
