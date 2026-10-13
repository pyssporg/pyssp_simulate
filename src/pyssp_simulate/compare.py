"""Comparison of simulation results across engines.

The entry point is :func:`compare`, which measures every result against
a reference result (the first one). Candidate signals are resampled
onto the reference time grid by linear interpolation before per-signal
error metrics are computed.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyssp_simulate.result import SimulationResult


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
) -> PairReport:
    """Compare one candidate result against a reference result."""
    reference_backend = reference.backend
    candidate_backend = candidate.backend

    common = [s for s in reference.signals if s in set(candidate.signals)]
    selected = list(signals) if signals is not None else common
    missing = [s for s in selected if s not in set(common)]
    if missing:
        raise KeyError(f"Signals missing from one of the results: {missing}")

    target = reference.time
    comparisons: list[SignalComparison] = []
    for signal in selected:
        reference_values = reference[signal]
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
    results: Sequence[SimulationResult], signals: Sequence[str] | None = None
) -> ComparisonReport:
    """Compare every result against the first one (the reference).

    :param results: two or more simulation results of the same SSP package.
    :param signals: optional subset of signals to compare (default: the
        signals present in every result).
    """
    if len(results) < 2:
        raise ValueError("Comparison needs at least two results")

    common = set(results[0].signals)
    for result in results[1:]:
        common &= set(result.signals)

    selected = list(signals) if signals is not None else sorted(common)
    pairs = tuple(compare_pair(results[0], result, selected) for result in results[1:])
    return ComparisonReport(
        results=tuple(results),
        pairs=pairs,
        common_signals=tuple(selected),
    )
