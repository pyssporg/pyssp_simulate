# Test fixtures

Committed test data lives one directory per case:

```
tests/fixtures/<case>/input.<ext>
tests/fixtures/<case>/case.md
```

`case.md` records the provenance, the fault the case pins, the independent
oracle, and the update policy; `tests/test_fixture_policy.py` enforces the
layout. No loose data files sit directly under this directory, and the
recording is never regenerated automatically — `tests/support/fixture_guard.py`
fails the session if a test modifies it.

## Cases

- [`omsimulator-recording/`](omsimulator-recording/case.md) — verbatim
  OMSimulator 3.0.0 result header and first rows for the
  `RampToIntegrator` fixture package; pins fake/real header agreement in
  `tests/test_backends.py`.

## Not fixtures

`tests/fmus/` holds the committed Modelica sources (`TestFMUs.mo`,
`build_fmus.mos`) for the FMUs compiled at test time. The compiled FMUs are
generated into a temporary or cache directory by `tests/generate_fmus.py`,
never committed here. See `tests/README.md` for the full layout.
