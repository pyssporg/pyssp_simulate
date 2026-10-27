---
id: INFO-002
type: info
title: Architecture
summary: "How pyssp_simulate is organized: runtime preparation to backend adapter to SimulationResult, with standard-artifact boundaries and explicit backend selection"
date: 2026-10-23
status: current
---

# Architecture

## Shape

```text
SimulationSpec
      │
      ▼
runtime preparation ──► backend adapter ──► SimulationResult
      │                       │                    │
      │                       │                    └─ result CSV
      └─ normalized SSP archive/root
```

## Boundaries

- `pyssp_simulate` consumes and produces standard SSP and result CSV files.
- A consumer receives only the public API: `SimulationSpec`,
  `SimulationResult`, `RuntimeSSP`, `PermutationConfig`, and the backend
  registry.
- Engine adapters are internal. They may import other pyssp projects only
  through public modules (`pyssp_vv.measure` for permutation reporting).
- Measurement and judgment never cross into this package. `pyssp_vv.judge` is
  not imported or re-exported.
- Backend selection is explicit. `simulate` runs the named backend;
  `DEFAULT_BACKEND` applies only when the caller names none. An engine that is
  missing raises `BackendNotAvailableError`; only `simulate_all` / `run-all`
  skip unavailable backends, and they warn.
- Result columns are canonical: `"time"` first, then signal names sorted, so a
  column shared by two engines has the same name and the same relative
  position. Engine-specific extra columns are allowed; engine-specific values
  are never substituted for a value the caller did not give.
