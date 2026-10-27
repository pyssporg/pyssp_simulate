# 02-architecture — Architecture

How pyssp_simulate is organized: the simulation pipeline, public surface, engine boundary, and structural rules.

Answers: "How is the system organized?"

Read when: you add or change a backend, a public type, or a boundary between this package and a sibling.

## Boundary

- Holds the organizing design only.
- Not specific files or scripts: absent `03-implementation/`.
- Not the proof strategy: the test suite is the proof; absent `04-verification/`.
- Measurement and judgment never cross into this package; the boundary is recorded in `AD-001`.

## Contents

<!-- pb:index:start -->
<!-- GENERATED FILE — do not edit. Regenerate with pb-registers. -->
- **INFO-002** [Architecture](architecture.md) — How pyssp_simulate is organized: runtime preparation to backend adapter to SimulationResult, with standard-artifact boundaries and explicit backend selection
<!-- pb:index:end -->
