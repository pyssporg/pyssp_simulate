# Case: `omsimulator-recording`

One file per fixture directory. It records what the bytes cannot: why the case
exists, what oracle proves it, and how the output may be normalized.

- **Conversion:** none — a raw engine result file, replayed verbatim
- **Requirement / fault:** the stubbed OMSimulator must emit the same column
  shape as the real engine; a hand-invented header must fail
  `test_omsimulator_stub_matches_recorded_header`
- **Level:** contract
- **Input:** `input.csv` — verbatim excerpt of the real OMSimulator result
  file for the fixture package `RampToIntegrator.ssp` (built at test time by
  `tests/conftest.py`): the header OMSimulator writes plus its first three
  data rows
- **Expected:** none — the input is its own oracle; the contract is the column
  set it defines
- **Independent oracle:** `read_result_csv(path, root_system=CONTAINER_NAMES)`
  from `pyssp_simulate.result`, then equality of the dtype names with the
  columns produced by the stubbed backend
- **Comparison:** parsed structure (dtype names), not text
- **Normalization:** none
- **Volatile fields:** none — this is a frozen recording
- **Update policy:** re-record by hand when the engine version or the fixture
  naming changes, review the diff, and update the record below; never
  auto-update
- **Notes:** recorded with OMSimulator 3.0.0 (`OMSimulator` Python package),
  `start_time=0.0`, `stop_time=1.0`, `step_size=0.001`,
  `omsimulator_options=("--emitEvents=false",)`, SSD name `RampToIntegrator`,
  system name `system`. Capture: run the backend with a persistent
  `working_dir` and copy the header plus the first rows of
  `<working_dir>/result.csv`. Do not canonicalize the header by hand: the
  quoted names, the `system.` container prefix, and the `_D_outputAlias` /
  derivative columns are the point of the case.

## What this case proves

The fake OMSimulator in `tests/test_backends.py` writes a real engine's result
header, so the adapter tests exercise the column shape OMSimulator actually
emits instead of a simplified `time,y` header no engine produces. If the case
is deleted or hand-edited into a canonical form, the stub and the real backend
can drift apart without any test noticing, and the adapter's column handling is
never checked against the engine.

## Update history

| Date | Change | Reason |
|---|---|---|
| 2026-10-28 | Moved from `tests/fixtures/omsimulator_result.csv`; provenance recorded in `case.md` | Give the recording an explicit case directory and oracle |
