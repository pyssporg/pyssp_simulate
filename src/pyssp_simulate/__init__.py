"""pyssp_simulate: simulate SSP packages with different engines.

Public API:

* :class:`SimulationSpec` -- what to simulate.
* :func:`simulate` / :func:`simulate_all` -- run one or all backends.
* :class:`SimulationResult` -- normalized result (structured NumPy array),
  with CSV helpers :func:`read_result_csv` / :func:`write_result_csv` and
  :func:`canonicalize_signal`.
* :class:`PermutationConfig` / :func:`load_permutation_config` /
  :func:`iter_permutation_specs` / :func:`run_permutations` /
  :class:`PermutationRun` -- batch sweeps described by a JSON config.
* :func:`available_backends` / :func:`backend_names` / :func:`get_simulator` /
  :class:`Simulator` / :class:`BackendNotAvailableError` /
  ``BACKEND_MODULES`` -- backend registry.
* :func:`spec_summary` -- plain-dict view of a spec.

Measurement and judgment are not part of this package. They live in
``pyssp_vv``: ``pyssp_vv.measure`` for trajectory comparison and
``pyssp_vv.judge`` for verdicts. This package imports ``pyssp_vv.measure``
internally for permutation runs and does not re-export it.

Example::

    from pyssp_simulate import SimulationSpec, simulate_all

    spec = SimulationSpec(ssp="model.ssp", start_time=0.0, stop_time=1.0,
                          step_size=0.001)
    results = simulate_all(spec)          # every backend that is installed
    for name, result in results.items():
        print(name, len(result.signals), len(result))
"""

from pyssp_simulate.interface import (
    BACKEND_MODULES,
    BackendNotAvailableError,
    Simulator,
    available_backends,
    backend_names,
    get_simulator,
    simulate,
    simulate_all,
)
from pyssp_simulate.permutations import (
    PermutationConfig,
    PermutationRun,
    iter_permutation_specs,
    load_permutation_config,
    run_permutations,
)
from pyssp_simulate.result import (
    TIME_SIGNAL,
    SimulationResult,
    canonicalize_signal,
    read_result_csv,
    write_result_csv,
)
from pyssp_simulate.spec import SimulationSpec, spec_summary

__version__ = "0.2.0"

__all__ = [
    "BACKEND_MODULES",
    "BackendNotAvailableError",
    "PermutationConfig",
    "PermutationRun",
    "SimulationResult",
    "SimulationSpec",
    "Simulator",
    "TIME_SIGNAL",
    "available_backends",
    "backend_names",
    "canonicalize_signal",
    "get_simulator",
    "iter_permutation_specs",
    "load_permutation_config",
    "read_result_csv",
    "run_permutations",
    "simulate",
    "simulate_all",
    "spec_summary",
    "write_result_csv",
]
