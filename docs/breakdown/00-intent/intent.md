---
id: INFO-001
type: info
title: Intent
summary: "Why pyssp_simulate exists: run SSP packages through pluggable engines and normalise results; measurement and judgment stay in pyssp_vv"
date: 2026-10-22
status: current
---

# Intent

## Purpose

`pyssp_simulate` runs SSP packages through pluggable simulation engines and
normalises the result into a structured trajectory. It owns engine execution
and result shape; it does not own measurement or judgment.

## Scope

| Belongs here | Belongs elsewhere |
| --- | --- |
| engine adapters and backend registry | metrics and criteria (`pyssp_vv`) |
| SSP runtime preparation | SSP XML and archive editing (`pyssp_standard`) |
| normalized result CSV and `SimulationResult` | workflow scheduling (`pyssp_workflow`) |
| permutation runs over simulation specs | SysML generation (`pyssp_sysml2`) |

## Commitments

- **Standard artifacts first at library borders.** Prefer the SSP package,
  result CSV, and other standard files. When an object must cross, it is part
  of the documented public API: `SimulationSpec`, `SimulationResult`,
  `RuntimeSSP`, `PermutationConfig`, the backend registry, and the result CSV
  helpers (`read_result_csv`, `write_result_csv`, `canonicalize_signal`).
- **No internal modules cross.** Consumers import the package public surface,
  not `pyssp_simulate.backends` internals or private helpers.
- **Simulation only.** This package never returns an acceptance decision.
- **Engine failure is explicit.** A missing backend or failed engine raises or
  records an error; it does not silently fall back to a different engine.

## Non-goals

- Measurement, judgment, or evidence.
- SSP XML editing.
- Workflow composition, retries, or campaign scheduling.
