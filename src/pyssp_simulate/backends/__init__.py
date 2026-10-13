"""Simulation backends.

Each module in this package wraps one simulation engine and exposes a
``SIMULATOR`` attribute (a :class:`pyssp_simulate.interface.Simulator`
subclass). The modules are imported lazily so that missing engines never
break imports of :mod:`pyssp_simulate`.
"""

from pyssp_simulate.interface import (  # noqa: F401
    BACKEND_MODULES,
    BackendNotAvailableError,
    Simulator,
    available_backends,
    backend_names,
    get_simulator,
)

__all__ = [
    "BACKEND_MODULES",
    "BackendNotAvailableError",
    "Simulator",
    "available_backends",
    "backend_names",
    "get_simulator",
]
