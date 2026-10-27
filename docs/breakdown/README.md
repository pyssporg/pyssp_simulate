# Product Breakdown — pyssp_simulate

Current-state record for pyssp_simulate: why it exists, what it commits to, how it is organized, and how it is checked. For authors and agents.

## The Seven Layers

- `00-intent/` — why does the project exist, who is it for?
- `01-product/` — absent: no separate capability leaves; the commitments in `INFO-001` carry the promise.
- `02-architecture/` — how are deliverables, evidence, and work organized?
- `03-implementation/` — absent: code-level facts live in the package README and docstrings.
- `04-verification/` — absent: the test suite is the proof; it has no leaf here.
- `05-operation/` — absent: install, run, and release live in the repository README.
- `06-evolution/` — absent: no open candidates.
- the decision archive (`docs/archive/archive.zip`) — the flat, dated decision stream: history, never current state; read via `pb archive`, never browsed

## Boundary Rule

A document's home is the layer whose question it answers; information flows downward only.
- Current state: the layer leaves, present tense.
- History: the archive (`docs/archive/archive.zip`), one dated record per committed choice.
- Tracking: the generated registers in `docs/archive/` (`README.md`, `design-choice-log.md`, `traceability-map.md`).
- This file indexes current state only.

## Reading Order

1. This file, then the layer index that owns the concern.
2. Open the one leaf whose summary matches; stop when the fact is found.
3. History: read one archive record by ID via `pb archive read`; never browse the stream.

## Route an Edit

- Current-state fact: update the owning layer leaf.
- Dated choice: a record via `pb archive add` into `docs/archive/archive.zip`.
- Candidate change: an IMP in `06-evolution/selected/`, once that layer exists.
- One fact has exactly one home; cite nodes by ID, never by path.
- Never hand-edit generated registers.

## Excludes

- Implementation detail and durable rationale: `docs/` and the package README, outside this tree.
