"""Paths to committed test fixtures.

Every fixture is a case directory with an ``input.<ext>`` file and a
``case.md`` (see ``tests/fixtures/README.md``); tests import the path from
here instead of building one from ``__file__`` or the working directory.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Verbatim OMSimulator result excerpt for the ``RampToIntegrator`` package;
#: see ``tests/fixtures/omsimulator-recording/case.md``.
OMSIMULATOR_RECORDING = FIXTURES / "omsimulator-recording" / "input.csv"
