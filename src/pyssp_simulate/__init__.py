"""pyssp_simulate: simulate SSP packages with different engines, compare results.

Public API:

* :class:`SimulationSpec` -- what to simulate.
* :func:`simulate` / :func:`simulate_all` -- run one or more backends.
* :class:`SimulationResult` -- normalized result (structured NumPy array).
* :func:`compare` / :class:`ComparisonReport` -- cross-engine comparison.
* :func:`available_backends` / :func:`get_simulator` -- backend registry.
* :class:`PermutationConfig` / :func:`run_permutations` -- batch sweeps
  described by a JSON config.

Example::

    from pyssp_simulate import SimulationSpec, simulate_all, compare

    spec = SimulationSpec(ssp="model.ssp", start_time=0.0, stop_time=10.0)
    results = simulate_all(spec)          # every backend that is installed
    report = compare(results)             # first result is the reference
    print(report.summary())
"""

from pyssp_simulate.compare import (
    ComparisonReport,
    PairReport,
    SignalComparison,
    compare,
    compare_pair,
)
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
    ComparisonSettings,
    PermutationConfig,
    PermutationRun,
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

__version__ = "0.1.0"

__all__ = [
    "BACKEND_MODULES",
    "BackendNotAvailableError",
    "ComparisonReport",
    "ComparisonSettings",
    "PairReport",
    "PermutationConfig",
    "PermutationRun",
    "SimulationResult",
    "SimulationSpec",
    "SignalComparison",
    "Simulator",
    "TIME_SIGNAL",
    "available_backends",
    "backend_names",
    "canonicalize_signal",
    "compare",
    "compare_pair",
    "get_simulator",
    "load_permutation_config",
    "read_result_csv",
    "run_permutations",
    "simulate",
    "simulate_all",
    "spec_summary",
    "write_result_csv",
]
