"""Comparison of simulation results across engines.

The entry point is :func:`compare`, which measures every result against
a reference result (the first one). Candidate signals are resampled
onto the reference time grid by linear interpolation before per-signal
error metrics are computed.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyssp_simulate.result import SimulationResult


@dataclass(frozen=True)
class ComparisonWindow:
    """Explicit time grid for comparison.

    Use this when every result must be judged on the same grid, independent of
    whichever result happens to be the reference.
    """

    start_time: float
    stop_time: float
    step_size: float

    def __post_init__(self) -> None:
        if self.stop_time <= self.start_time:
            raise ValueError("stop_time must be greater than start_time")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

    def time_grid(self) -> np.ndarray:
        span = self.stop_time - self.start_time
        steps = int(round(span / self.step_size))
        return np.linspace(
            self.start_time, self.stop_time, steps + 1, dtype=np.float64
        )


@dataclass(frozen=True)
class SignalComparison:
    """Error metrics of one candidate result vs the reference, one signal."""

    signal: str
    reference_backend: str
    candidate_backend: str
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    reference_min: float
    reference_max: float
    candidate_min: float
    candidate_max: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "signal": self.signal,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "reference_min": self.reference_min,
            "reference_max": self.reference_max,
            "candidate_min": self.candidate_min,
            "candidate_max": self.candidate_max,
        }


@dataclass(frozen=True)
class PairReport:
    """All signal comparisons between one reference and one candidate."""

    reference_backend: str
    candidate_backend: str
    comparisons: tuple[SignalComparison, ...]

    @property
    def max_abs_error(self) -> float:
        return max((item.max_abs_error for item in self.comparisons), default=0.0)

    @property
    def mean_abs_error(self) -> float:
        values = [item.mean_abs_error for item in self.comparisons]
        return float(np.mean(values)) if values else 0.0

    @property
    def rmse(self) -> float:
        values = [item.rmse for item in self.comparisons]
        return float(np.mean(values)) if values else 0.0

    def passed(self, atol: float = 1e-6, rtol: float = 0.0) -> bool:
        """Whether every signal's ``max_abs_error`` is within tolerance.

        A signal passes when ``max_abs_error <= atol + rtol * max(
        |reference_min|, |reference_max|)`` — the tolerance is scaled by
        the reference's peak absolute value, not its range.
        """
        return all(
            item.max_abs_error
            <= atol + rtol * max(abs(item.reference_max), abs(item.reference_min))
            for item in self.comparisons
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "signal_count": len(self.comparisons),
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Result of comparing several simulation results against a reference."""

    results: tuple[SimulationResult, ...]
    pairs: tuple[PairReport, ...]
    common_signals: tuple[str, ...]

    def passed(self, atol: float = 1e-6, rtol: float = 0.0) -> bool:
        """Whether every pair passed (see :meth:`PairReport.passed`)."""
        return all(pair.passed(atol, rtol) for pair in self.pairs)

    def summary(self) -> dict:
        return {
            "backends": [result.backend for result in self.results],
            "common_signals": list(self.common_signals),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "max_abs_error": max((pair.max_abs_error for pair in self.pairs), default=0.0),
        }

    def write_csv(self, path: Path | str) -> Path:
        """Write the per-signal metrics of every pair to one CSV."""
        from dataclasses import fields

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [field.name for field in fields(SignalComparison)]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for pair in self.pairs:
                for item in pair.comparisons:
                    writer.writerow(item.to_dict())
        return path


def resample(times: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Linearly resample ``values`` onto ``target`` times.

    Non-finite samples are dropped; a single valid sample becomes a
    constant series; out-of-range targets are clamped to the end values.
    """
    valid = np.isfinite(times) & np.isfinite(values)
    valid_times = times[valid]
    valid_values = values[valid]
    if len(valid_times) == 0:
        return np.full(target.shape, np.nan, dtype=np.float64)

    order = np.argsort(valid_times)
    valid_times = valid_times[order]
    valid_values = valid_values[order]
    unique_times, unique_index = np.unique(valid_times, return_index=True)
    unique_values = valid_values[unique_index]

    if len(unique_times) == 1:
        return np.full(target.shape, unique_values[0], dtype=np.float64)

    return np.interp(
        target,
        unique_times,
        unique_values,
        left=unique_values[0],
        right=unique_values[-1],
    )


def compare_pair(
    reference: SimulationResult,
    candidate: SimulationResult,
    signals: Sequence[str] | None = None,
    window: ComparisonWindow | None = None,
) -> PairReport:
    """Compare one candidate result against a reference result.

    :param window: optional explicit time grid; when omitted the reference
        result's own time samples are used.
    """
    reference_backend = reference.backend
    candidate_backend = candidate.backend

    common = [s for s in reference.signals if s in set(candidate.signals)]
    selected = list(signals) if signals is not None else common
    missing = [s for s in selected if s not in set(common)]
    if missing:
        raise KeyError(f"Signals missing from one of the results: {missing}")

    target = window.time_grid() if window is not None else reference.time
    comparisons: list[SignalComparison] = []
    for signal in selected:
        reference_values = resample(reference.time, reference[signal], target)
        candidate_values = resample(candidate.time, candidate[signal], target)
        errors = np.abs(reference_values - candidate_values)
        comparisons.append(
            SignalComparison(
                signal=signal,
                reference_backend=reference_backend,
                candidate_backend=candidate_backend,
                max_abs_error=float(np.nanmax(errors)),
                mean_abs_error=float(np.nanmean(errors)),
                rmse=float(np.sqrt(np.nanmean(np.square(errors)))),
                reference_min=float(np.nanmin(reference_values)),
                reference_max=float(np.nanmax(reference_values)),
                candidate_min=float(np.nanmin(candidate_values)),
                candidate_max=float(np.nanmax(candidate_values)),
            )
        )
    return PairReport(
        reference_backend=reference_backend,
        candidate_backend=candidate_backend,
        comparisons=tuple(comparisons),
    )


def compare(
    results: Sequence[SimulationResult] | Mapping[str, SimulationResult],
    signals: Sequence[str] | None = None,
    window: ComparisonWindow | None = None,
) -> ComparisonReport:
    """Compare every result against the first one (the reference).

    :param results: two or more simulation results of the same SSP
        package. Accepts a sequence, or the mapping returned by
        :func:`~pyssp_simulate.simulate_all` (its values are used, in
        the order the backends ran).
    :param signals: optional subset of signals to compare (default: the
        signals present in every result).
    :param window: optional explicit time grid (see :class:`ComparisonWindow`).
    :raises ValueError: if fewer than two results are given.
    :raises KeyError: if a requested signal is missing from a result.
    """
    if isinstance(results, Mapping):
        results = list(results.values())
    else:
        results = list(results)

    if len(results) < 2:
        raise ValueError("Comparison needs at least two results")

    common = set(results[0].signals)
    for result in results[1:]:
        common &= set(result.signals)

    selected = list(signals) if signals is not None else sorted(common)
    pairs = tuple(
        compare_pair(results[0], result, selected, window=window)
        for result in results[1:]
    )
    return ComparisonReport(
        results=tuple(results),
        pairs=pairs,
        common_signals=tuple(selected),
    )


def compare_to_reference(
    reference_path: Path | str,
    results: Sequence[SimulationResult] | Mapping[str, SimulationResult],
    *,
    model_name: str | None = None,
    prefix: str | None = None,
    reference_backend: str = "reference",
    signals: Sequence[str] | None = None,
    window: ComparisonWindow | None = None,
) -> ComparisonReport:
    """Compare one or more results against a trusted reference CSV.

    The reference CSV is loaded with
    :meth:`SimulationResult.from_reference_csv`, then every result is compared
    against it using :func:`compare`.
    """
    reference = SimulationResult.from_reference_csv(
        reference_path,
        model_name=model_name,
        prefix=prefix,
        backend=reference_backend,
    )
    candidates = (
        list(results.values()) if isinstance(results, Mapping) else list(results)
    )
    if not candidates:
        raise ValueError("Comparison needs at least one candidate result")
    return compare([reference, *candidates], signals=signals, window=window)
