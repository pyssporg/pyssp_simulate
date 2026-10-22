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
