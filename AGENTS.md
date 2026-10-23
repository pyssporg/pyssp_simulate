# AGENTS

## Vision first (required)

Before you read code, edit files, or plan a change, read the project vision:

**`docs/breakdown/00-intent/intent.md`**

It states why this repository exists, what it commits to, and what it refuses to
become. It is the first document to read, not a reference of last resort.

- Read it in full before making any change to code, tests, docs, or artifacts.
- Judge every proposed change against it. If a change conflicts with the vision,
  its commitments, goals, or non-goals, stop and surface the conflict; do not
  implement it silently.
- If the work shows the long-term direction is wrong, change the vision
  deliberately first, then the code. Never let the repository drift away from
  its stated intent.

## Rules

1. **Don't assume. Surface tradeoffs.** If a contract is ambiguous, stop and say
   so rather than picking a behaviour silently.
2. **Minimum code that solves the problem.** Nothing speculative.
3. **Touch only what you must.** Clean up only your own mess.
4. **Define success criteria and verify.**

## Repository Focus

`pyssp_simulate` runs SSP packages through pluggable simulation engines and
normalises the result into a structured trajectory. It owns engine execution
and result shape; measurement and judgment stay in `pyssp_vv`.

- Engine adapters live in `src/pyssp_simulate/backends/` and are internal.
- The package `__init__` is the public surface; `pyssp-simulate` is the CLI.

## Hard constraints

- **Simulation only.** This package runs engines and normalises their output.
  It never returns an acceptance decision, and it never imports or re-exports
  `pyssp_vv.judge`.
- **Standard artifacts first at library borders.** Prefer the SSP package and
  result CSV. When an object crosses, it is part of the documented public API:
  `SimulationSpec`, `SimulationResult`, `RuntimeSSP`, `PermutationConfig`, the
  backend registry, and the result CSV helpers.
- **Engine adapters are internal.** Consumers import the package public surface,
  not `pyssp_simulate.backends` internals or private helpers.
- **Cross-project imports use public modules only.** The one exception is
  `pyssp_vv.measure`, imported locally for permutation reporting and never
  re-exported.
- **Engine failure is explicit.** A missing backend or failed engine raises or
  records an error; it never silently falls back to a different engine.
- **Inputs are read-only.** Runtime preparation writes a normalised copy; it
  does not edit the caller's SSP.

## Code guidelines

- Keep engine differences behind the backend interface; callers see one result
  shape.
- Prefer explicit specs and plain result data over ambient state.
- Avoid abstractions that hide which engine ran or how a result was produced.

## Conventions

- `docs/breakdown/` holds intent and architecture.
- Architecture decisions are `docs/breakdown/02-architecture/decisions/ADR-<NNN>.md`.
- When a decision changes the boundary, update the intent or architecture leaf
  rather than leaving the code as the only record.

## Environment

- Python >= 3.10. Backends whose engines are not installed are skipped
  automatically.
- Run the tests with `python -m pytest -q` (use a repo-local `venv` when
  present); `ruff check src tests` for lint.
- Integration tests need `omc` on `PATH` to build the two primitive FMUs; they
  skip without it. `python tests/generate_fmus.py` prepares the FMUs once.
