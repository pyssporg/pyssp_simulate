"""The backend interface and the public simulation entry points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.util import find_spec
from typing import ClassVar

from pyssp_simulate.result import SimulationResult
from pyssp_simulate.spec import SimulationSpec

# Backend name -> module providing a ``SIMULATOR`` attribute.
# Kept as strings so importing pyssp_simulate never imports an engine.
BACKEND_MODULES: dict[str, str] = {
    "fmpy": "pyssp_simulate.backends.fmpy_backend",
    "omsimulator": "pyssp_simulate.backends.omsimulator_backend",
    "ssp4sim": "pyssp_simulate.backends.ssp4sim_backend",
    "cosim": "pyssp_simulate.backends.cosim_backend",
}

# Order used when backend="auto".
BACKEND_PRIORITY = ("fmpy", "omsimulator", "ssp4sim", "cosim")


class BackendNotAvailableError(RuntimeError):
    """Raised when a backend is requested but its engine is not installed."""


class Simulator(ABC):
    """Base class for simulation backends.

    Subclasses live in :mod:`pyssp_simulate.backends` and are looked up by
    their :attr:`name` through the registry in this module.
    """

    #: Registry key and CLI value for this backend.
    name: ClassVar[str]
    #: How to install the engine, shown in error messages.
    install_hint: ClassVar[str] = ""

    @classmethod
    @abstractmethod
    def available(cls) -> tuple[bool, str | None]:
        """Return ``(is_available, reason_if_not)``."""

    @abstractmethod
    def simulate(self, spec: SimulationSpec) -> SimulationResult:
        """Run the simulation described by ``spec``."""


def import_module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def get_simulator_class(name: str) -> type[Simulator]:
    """Resolve a backend name to its :class:`Simulator` subclass."""
    key = name.lower()
    if key not in BACKEND_MODULES:
        raise KeyError(
            f"Unknown backend {name!r}; available backends: {backend_names()}"
        )
    import importlib

    module = importlib.import_module(BACKEND_MODULES[key])
    return module.SIMULATOR


def get_simulator(name: str) -> Simulator:
    """Instantiate a backend by name, raising if its engine is missing."""
    key = name.lower()
    simulator_class = get_simulator_class(key)
    ok, reason = simulator_class.available()
    if not ok:
        raise BackendNotAvailableError(
            f"Backend {key!r} is not available: {reason}. {simulator_class.install_hint}"
        )
    return simulator_class()


def backend_names() -> list[str]:
    return list(BACKEND_MODULES)


def available_backends() -> dict[str, tuple[bool, str | None]]:
    """Check every registered backend without importing the engines' modules."""
    status: dict[str, tuple[bool, str | None]] = {}
    for name in BACKEND_MODULES:
        try:
            simulator_class = get_simulator_class(name)
        except Exception as error:  # pragma: no cover - defensive
            status[name] = (False, f"failed to load backend module: {error}")
            continue
        status[name] = simulator_class.available()
    return status


def simulate(spec: SimulationSpec, backend: str = "auto") -> SimulationResult:
    """Simulate an SSP package with one backend.

    :param spec: what to simulate (see :class:`SimulationSpec`).
    :param backend: backend name, or ``"auto"`` for the first available
        backend in registry order.
    """
    if backend.lower() == "auto":
        for name in BACKEND_PRIORITY:
            simulator_class = get_simulator_class(name)
            ok, _ = simulator_class.available()
            if ok:
                backend = name
                break
        else:
            raise BackendNotAvailableError(
                "No simulation backend is available. Install at least one "
                "engine, e.g. `pip install fmpy`."
            )

    simulator = get_simulator(backend)
    return simulator.simulate(spec)


def simulate_all(
    spec: SimulationSpec, backends: list[str] | None = None
) -> dict[str, SimulationResult]:
    """Simulate an SSP package with several backends and collect the results.

    Runs every backend in ``backends`` (default: all registered) whose
    engine is installed, skipping the others. Raises when no backend ran.
    """
    names = backends if backends is not None else backend_names()

    results: dict[str, SimulationResult] = {}
    for name in names:
        simulator_class = get_simulator_class(name)
        ok, reason = simulator_class.available()
        if not ok:
            import warnings

            warnings.warn(
                f"Skipping backend {name!r}: {reason}", RuntimeWarning, stacklevel=2
            )
            continue
        results[name] = simulator_class().simulate(spec)

    if not results:
        raise BackendNotAvailableError(
            "None of the requested backends are available: " + ", ".join(names)
        )
    return results
