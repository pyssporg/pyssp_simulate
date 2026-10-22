"""Batch execution of simulation permutations described by a JSON config.

A *permutation config* is a JSON object naming one SSP package and the
sets of simulation parameters to sweep. Each numeric sweep field
(``start_time``, ``stop_time``, ``step_size``, ``tolerance``) accepts a
scalar, a list of values, or ``null``; the executed runs are the cartesian
product of all the sets, each executed with every configured backend.
Whenever two or more backends produce a result, the results are measured
against the first backend that ran (in configured order). Judgment is not
performed here: verdicts belong to ``pyssp_vv.judge``.

Full schema -- only ``ssp`` is required, every other key is optional::

    {
        "ssp": "model.ssp",
        "backends": ["fmpy", "omsimulator"],
        "start_time": 0.0,
        "stop_time": [0.05, 0.1],
        "step_size": [0.001, null],
        "tolerance": null,
        "output_dir": "results"
    }

Key semantics:

* ``ssp`` (required): path to an ``.ssp`` archive or an unpacked SSP
  directory. Existence and shape are validated by
  :class:`~pyssp_simulate.spec.SimulationSpec` when the specs are built.
* ``backends``: list of backend names, or ``null`` for every registered
  backend. Backends whose engine is not installed are skipped and the
  reason is recorded on the run.
* ``start_time`` / ``stop_time``: scalar, list, or ``null`` for the
  default (``0.0`` / ``1.0``).
* ``step_size`` / ``tolerance``: scalar, list, or ``null``; ``null`` (and
  ``null`` list entries) let each backend pick its own default.
* ``output_dir``: directory for the run artifacts, or ``null`` to keep
  everything in memory.

The example above describes 2 stop times x 2 step sizes = 4 permutations
for each of the two backends. With ``output_dir`` set, permutation ``i``
writes ``<output_dir>/perm_<i>/<backend>.csv`` for every backend that
ran, a ``<output_dir>/perm_<i>/spec.json`` manifest and -- when two or
more backends produced results and ``pyssp-vv[measure]`` is installed --
the cross-backend metrics as ``<output_dir>/perm_<i>/metrics.csv``. Indices
are formatted with four digits (``perm_0000``, ``perm_0001``, ...).

Unknown keys are rejected with :class:`ValueError` (typo protection). An
empty set (for example ``"stop_time": []``) produces zero permutations;
:func:`run_permutations` returns an empty list and callers decide how to
treat that.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from pyssp_simulate.interface import (
    BackendNotAvailableError,
    backend_names,
    get_simulator_class,
)
from pyssp_simulate.result import SimulationResult
from pyssp_simulate.spec import SimulationSpec, spec_summary

if TYPE_CHECKING:
    from pyssp_vv.measure import Comparison

#: Keys accepted at the top level of a permutation config JSON file.
_CONFIG_KEYS = frozenset(
    {
        "ssp",
        "backends",
        "start_time",
        "stop_time",
        "step_size",
        "tolerance",
        "output_dir",
    }
)


@dataclass(frozen=True)
class PermutationConfig:
    """A normalized permutation sweep over one SSP package.

    :param ssp: path to the SSP archive or unpacked SSP directory.
    :param backends: backend names to run, or ``None`` for every
        registered backend.
    :param start_times: start times to sweep.
    :param stop_times: stop times to sweep.
    :param step_sizes: step sizes to sweep; ``None`` entries let each
        backend pick its own default.
    :param tolerances: solver/iteration tolerances to sweep; ``None``
        entries let each backend pick its own default.
    :param output_dir: directory for the per-permutation artifacts, or
        ``None`` to keep everything in memory.

    The executed runs are the cartesian product of the four sweep sets
    (see :func:`iter_permutation_specs`); an empty set yields no runs.
    """

    ssp: Path
    backends: tuple[str, ...] | None
    start_times: tuple[float, ...] = (0.0,)
    stop_times: tuple[float, ...] = (1.0,)
    step_sizes: tuple[float | None, ...] = (None,)
    tolerances: tuple[float | None, ...] = (None,)
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ssp", Path(self.ssp))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))


def _as_float(value: object, key: str) -> float:
    """Coerce one JSON value to ``float``, rejecting anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must contain only numbers, got {value!r}")
    return float(value)


def _float_values(raw: object, key: str, default: float) -> tuple[float, ...]:
    """Normalize a scalar/list/null JSON field to a tuple of floats."""
    if raw is None:
        return (default,)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return (float(raw),)
    if isinstance(raw, list):
        return tuple(_as_float(value, key) for value in raw)
    raise ValueError(f"'{key}' must be a number, a list of numbers, or null")


def _optional_float_values(raw: object, key: str) -> tuple[float | None, ...]:
    """Normalize a scalar/list/null JSON field to floats with ``None`` entries."""
    if raw is None:
        return (None,)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return (float(raw),)
    if isinstance(raw, list):
        return tuple(None if value is None else _as_float(value, key) for value in raw)
    raise ValueError(f"'{key}' must be a number, a list, or null")


def _config_from_dict(data: dict) -> PermutationConfig:
    """Build a :class:`PermutationConfig` from a parsed JSON object."""
    unknown = sorted(set(data) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown permutation config key(s): {', '.join(unknown)}")

    if "ssp" not in data:
        raise ValueError("Permutation config is missing the required 'ssp' key")
    ssp = data["ssp"]
    if not isinstance(ssp, str):
        raise ValueError("'ssp' must be a string path")

    raw_backends = data.get("backends")
    if raw_backends is None:
        backends: tuple[str, ...] | None = None
    elif isinstance(raw_backends, list) and all(
        isinstance(name, str) for name in raw_backends
    ):
        backends = tuple(name.strip().lower() for name in raw_backends)
    else:
        raise ValueError("'backends' must be a list of backend names or null")

    raw_output_dir = data.get("output_dir")
    if raw_output_dir is None:
        output_dir: Path | None = None
    elif isinstance(raw_output_dir, str):
        output_dir = Path(raw_output_dir)
    else:
        raise ValueError("'output_dir' must be a string path or null")

    return PermutationConfig(
        ssp=Path(ssp),
        backends=backends,
        start_times=_float_values(data.get("start_time"), "start_time", 0.0),
        stop_times=_float_values(data.get("stop_time"), "stop_time", 1.0),
        step_sizes=_optional_float_values(data.get("step_size"), "step_size"),
        tolerances=_optional_float_values(data.get("tolerance"), "tolerance"),
        output_dir=output_dir,
    )


def load_permutation_config(path: Path | str) -> PermutationConfig:
    """Load and normalize a permutation config from a JSON file.

    :param path: path to the JSON config file.
    :raises ValueError: on unknown keys, missing ``ssp``, or values that
        do not match the schema documented in the module docstring.
    :raises FileNotFoundError: if ``path`` itself does not exist.
    """
    with Path(path).open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Permutation config {path} must be a JSON object")
    return _config_from_dict(data)


def iter_permutation_specs(config: PermutationConfig) -> Iterator[SimulationSpec]:
    """Yield one :class:`SimulationSpec` per cartesian permutation.

    The product runs over ``(start_times, stop_times, step_sizes,
    tolerances)`` with the last factor varying fastest, so the number of
    specs is the product of the set sizes (zero when any set is empty).
    Each spec is validated by ``SimulationSpec`` itself, which raises for
    a missing ``ssp`` path or an inconsistent parameter combination.
    """
    for start_time, stop_time, step_size, tolerance in product(
        config.start_times, config.stop_times, config.step_sizes, config.tolerances
    ):
        yield SimulationSpec(
            ssp=config.ssp,
            start_time=start_time,
            stop_time=stop_time,
            step_size=step_size,
            tolerance=tolerance,
        )


@dataclass
class PermutationRun:
    """The outcome of one permutation across the configured backends.

    :param spec: the simulation request that was run.
    :param results: backend name -> result, in configured backend order.
    :param measurements: one comparison per candidate backend against the
        first backend that ran, or ``None`` when fewer than two backends
        produced a result.
    :param output_dir: directory the run's artifacts were written to, or
        ``None`` when results are kept in memory only.
    :param skipped: backend name -> reason it could not run.
    """

    spec: SimulationSpec
    results: dict[str, SimulationResult]
    measurements: tuple[Comparison, ...] | None
    output_dir: Path | None
    skipped: dict[str, str]

    def summary(self) -> dict:
        """A compact, JSON-friendly description of the run."""
        return {
            "spec": spec_summary(self.spec),
            "backends": list(self.results),
            "skipped": dict(self.skipped),
            "measurements": (
                [item.to_dict() for item in self.measurements]
                if self.measurements is not None
                else None
            ),
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
        }


def _measure(results: dict[str, SimulationResult]) -> tuple[Comparison, ...]:
    """Measure every result against the first one, using ``pyssp_vv``.

    This is a convenience inside the package: the import is local so that
    ``pyssp_simulate`` installs and imports without the measurement extra. The
    surface is not re-exported, and judgment is not performed here.
    """
    try:
        from pyssp_vv.measure import Trajectory, compare_many
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "Cross-backend measurement requires 'pyssp-vv[measure]'. "
            "Install it, or run with a single backend."
        ) from exc

    trajectories = [
        Trajectory(
            time=result.time,
            channels={signal: result[signal] for signal in result.signals},
            label=name,
        )
        for name, result in results.items()
    ]
    return compare_many(trajectories)


def _write_metrics(path: Path, measurements: Sequence[Comparison]) -> Path:
    rows = [row for item in measurements for row in item.rows()]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["signal"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_permutation(
    index: int, spec: SimulationSpec, names: Sequence[str], config: PermutationConfig
) -> PermutationRun:
    """Run one spec with every backend in ``names`` and collect the outcome."""
    results: dict[str, SimulationResult] = {}
    skipped: dict[str, str] = {}
    for name in names:
        simulator_class = get_simulator_class(name)
        ok, reason = simulator_class.available()
        if not ok:
            skipped[name] = reason or f"backend {name!r} is not available"
            continue
        try:
            results[name] = simulator_class().simulate(spec)
        except BackendNotAvailableError as error:
            skipped[name] = str(error)

    measurements = _measure(results) if len(results) >= 2 else None

    run_dir: Path | None = None
    if config.output_dir is not None:
        run_dir = config.output_dir / f"perm_{index:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps(spec_summary(spec), indent=2))
        for name, result in results.items():
            result.write_csv(run_dir / f"{name}.csv")
        if measurements is not None:
            _write_metrics(run_dir / "metrics.csv", measurements)

    return PermutationRun(
        spec=spec,
        results=results,
        measurements=measurements,
        output_dir=run_dir,
        skipped=skipped,
    )


def run_permutations(config: PermutationConfig) -> list[PermutationRun]:
    """Run every permutation in the config, one :class:`PermutationRun` per spec.

    Backends are resolved in configured order (all registered backends
    when ``config.backends`` is ``None``). Unavailable backends are
    skipped with the reason recorded in ``run.skipped``; simulations that
    raise :class:`BackendNotAvailableError` are likewise recorded as
    skipped. Any other exception propagates. Returns ``[]`` when the
    cartesian product is empty.

    :param config: the normalized permutation config to execute.
    """
    names = list(config.backends) if config.backends is not None else backend_names()
    runs: list[PermutationRun] = []
    for index, spec in enumerate(iter_permutation_specs(config)):
        runs.append(_run_permutation(index, spec, names, config))
    return runs
