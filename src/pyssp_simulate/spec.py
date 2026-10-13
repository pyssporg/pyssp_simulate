"""Simulation input specification shared by all backends."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationSpec:
    """Description of one simulation run of an SSP package.

    :param ssp: path to an ``.ssp`` archive or to an unpacked SSP directory
        (a directory containing an ``.ssd`` file, optionally with unpacked
        FMU directories that are repackaged at runtime).
    :param start_time: simulation start time.
    :param stop_time: simulation stop time.
    :param step_size: communication/logging step size. ``None`` lets each
        backend pick its own default.
    :param tolerance: optional solver/iteration tolerance.
    :param result_file: optional path the result CSV is written to. When
        ``None``, the result is only kept in memory.
    :param working_dir: optional directory for backend scratch files. When
        ``None``, a temporary directory is used and removed afterwards.
    """

    ssp: Path
    start_time: float = 0.0
    stop_time: float = 1.0
    step_size: float | None = None
    tolerance: float | None = None
    result_file: Path | None = None
    working_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ssp", Path(self.ssp))
        if self.result_file is not None:
            object.__setattr__(self, "result_file", Path(self.result_file))
        if self.working_dir is not None:
            object.__setattr__(self, "working_dir", Path(self.working_dir))

        if not self.ssp.exists():
            raise FileNotFoundError(f"SSP path does not exist: {self.ssp}")
        if self.ssp.is_dir() and not _looks_like_ssp_dir(self.ssp):
            raise ValueError(
                f"Directory {self.ssp} does not look like an unpacked SSP "
                "(no .ssd file found)"
            )
        if self.stop_time <= self.start_time:
            raise ValueError(
                f"stop_time ({self.stop_time}) must be greater than start_time "
                f"({self.start_time})"
            )
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.tolerance is not None and self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")

    @property
    def duration(self) -> float:
        return self.stop_time - self.start_time

    def effective_step_size(self) -> float:
        """The step size to use, falling back to a backend-friendly default."""
        return self.step_size if self.step_size is not None else self.duration / 100.0

    @contextmanager
    def workspace(self) -> Iterator[Path]:
        """Yield a directory backends can scribble in.

        Uses ``working_dir`` when given (created if missing, not cleaned up),
        otherwise a temporary directory that is removed afterwards.
        """
        if self.working_dir is not None:
            self.working_dir.mkdir(parents=True, exist_ok=True)
            yield self.working_dir
            return
        with tempfile.TemporaryDirectory(prefix="pyssp_simulate_") as tmp:
            yield Path(tmp)


def _looks_like_ssp_dir(path: Path) -> bool:
    return any(path.glob("*.ssd")) or any(path.rglob("*.ssd"))


def spec_summary(spec: SimulationSpec) -> dict:
    """A plain-dict view of a spec, safe for logging and manifests."""
    return {
        "ssp": str(spec.ssp),
        "start_time": spec.start_time,
        "stop_time": spec.stop_time,
        "step_size": spec.step_size,
        "tolerance": spec.tolerance,
        "result_file": str(spec.result_file) if spec.result_file else None,
        "working_dir": str(spec.working_dir) if spec.working_dir else None,
    }
