# Tests

Run from the package root:

```bash
python -m pytest -q
```

## Layout

| Path | Contents |
|---|---|
| `tests/test_*.py` | the suite, one file per subject |
| `tests/conftest.py` | fixture SSP construction, level markers, session guard |
| `tests/support/fixtures.py` | paths to committed fixtures |
| `tests/support/fixture_guard.py` | session-start/finish fixture digest check |
| `tests/fixtures/<case>/` | one directory per committed fixture case |
| `tests/fmus/` | committed Modelica sources for the FMUs compiled at test time |

## Fixture policy

- Committed data lives in `tests/fixtures/<case>/input.<ext>` with a
  `case.md` beside it recording provenance, the fault the case pins, the
  independent oracle, and the update policy. No loose data files sit under
  `tests/fixtures/`, and no test builds a fixture path from `__file__` or the
  working directory — import it from `tests/support/fixtures.py`.
- `tests/test_fixture_policy.py` enforces the layout: every case has
  `input.*` and `case.md`, the required headings are present, no test source
  contains a working-directory fixture path, and fixture files stay under
  64 KB unless allowlisted with a reason.
- The committed recording is
  `tests/fixtures/omsimulator-recording/input.csv`: a verbatim OMSimulator
  3.0.0 result header plus the first rows for the `RampToIntegrator` package.
  To change it, re-record by hand from a real engine run, review the diff, and
  update `case.md`; it is never regenerated automatically.

## Guard

`tests/support/fixture_guard.py` digests `tests/fixtures/` and `tests/fmus/`
at session start and again at session finish (excluding `__pycache__`,
`*.pyc`, and the generated `tests/fmus/build/`). If a test changed either
tree, the session exits non-zero and lists the changed paths. A test that
needs to write must write into `tmp_path`.

## Levels

Every test carries exactly one level marker; `conftest.py` applies the file
default from `FILE_LEVELS`, and a test may override it.

| Level | Command |
|---|---|
| static | `python -m pytest -m level_static` |
| unit | `python -m pytest -m level_unit` |
| component | `python -m pytest -m level_component` |
| contract | `python -m pytest -m level_contract` |
| integration | `python -m pytest -m level_integration` |
| system | `python -m pytest -m level_system` |
| acceptance | `python -m pytest -m level_acceptance` |
| slow (opt-in) | `python -m pytest -m slow` |

The integration and system tests run engines over the fixture SSP. They need
`omc` to compile the two primitive FMUs; without it they skip themselves.

## FMU cache

The FMUs are built by `tests/generate_fmus.py` from `tests/fmus/TestFMUs.mo`
through `tests/fmus/build_fmus.mos`. The suite compiles them into a session
temporary directory; the script's own default output is `tests/fmus/build/`,
which is generated and therefore excluded from the guard. To reuse a prebuilt
pair:

```bash
python tests/generate_fmus.py            # writes the FMUs plus FMU_MANIFEST.json
python tests/generate_fmus.py --output DIR --force
export PYSSP_SIMULATE_TEST_FMU_DIR=/path/to/fmu-cache
```

The cache must carry `FMU_MANIFEST.json`; a missing or mismatched manifest
fails the suite instead of reusing stale FMUs.

## Adding a fixture case

1. Create `tests/fixtures/<case>/` and put the bytes in `input.<ext>`.
2. Copy `tests/fixtures/omsimulator-recording/case.md`, fill every metadata
   bullet, and state which defect escapes if the case is deleted.
3. Export the path from `tests/support/fixtures.py` and use that constant in
   the test.
4. Run `python -m pytest -m level_static` — the policy test checks the layout.
