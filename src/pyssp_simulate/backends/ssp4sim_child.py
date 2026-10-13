"""Child process entry point for the ssp4sim backend.

Runs one ssp4sim simulation from a JSON configuration file and exits.
Run as a bare script: ``python ssp4sim_child.py <config.json>``.

This module must be executed directly, *not* via ``python -m``, and it
must not import ``pyssp_simulate``, ``numpy``, or anything else that
transitively loads a dynamic ``libstdc++.so.6`` (numpy pulls one in via
OpenBLAS and libgfortran). The pyssp4sim wheels statically link libstdc++
while leaving its symbols interposable; when a dynamic copy is already
loaded, the two cross-wire and the process segfaults during
configuration parsing. Importing ``pyssp4sim`` first in a fresh
interpreter lets the static runtime bind to itself, which is reliable.

The engine also does not reliably return control to the interpreter
after ``simulate()``, which is why it runs in this dedicated process and
reports success or failure via its exit code.
"""

from __future__ import annotations

import sys
import traceback

_USAGE = "usage: python ssp4sim_child.py <config.json>"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(_USAGE, file=sys.stderr)
        return 2

    config_path = args[0]
    try:
        from pyssp4sim import Simulator

        simulator = Simulator(config_path)
        simulator.init()
        simulator.simulate()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return 1

    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
