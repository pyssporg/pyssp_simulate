# Test fixtures

## `omsimulator_result.csv`

A recorded excerpt of the OMSimulator result file for the fixture package
(`tests/conftest.py`, `RampToIntegrator.ssp`): the exact header OMSimulator
3.0.0 writes for the ramp-to-integrator chain, plus the first three data
rows, verbatim.

Purpose: the stubbed OMSimulator in `tests/test_backends.py` writes this
file instead of a hand-invented header, so the fake exercises the real
engine's column shape (quoted names, `system.` container prefix, and the
`_D_outputAlias` / derivative columns). `test_backends.py` asserts the
stubbed and the real backend produce the same canonical column set.

Recorded with:

- engine: OMSimulator 3.0.0 (`OMSimulator` Python package)
- fixture constants: `START_TIME=0.0`, `STOP_TIME=1.0`, `STEP_SIZE=0.001`
- SSD name `RampToIntegrator`, system name `system`
- capture: run the backend with a persistent `working_dir` and copy the
  header plus the first rows of `<working_dir>/result.csv`.

Do not canonicalize the header by hand: the point of the fixture is the
raw engine output. Regenerate it only when the recorded engine version or
the fixture naming changes, and record the new provenance here.
