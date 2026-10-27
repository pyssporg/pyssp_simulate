<!-- GENERATED FILE — do not edit. Regenerate with pb-registers. -->

# Decision Stream

Flat, dated history of every committed design choice. Current state lives
in the layer leaves named by each record's `state:` field.

Records live in the cold-storage archive; extract one with `pb archive read <ID>`.

## By Layer

### Architecture

- **AD-001** — Comparison moves to pyssp_vv · accepted · 2026-10-22 · `AD-001`
  Comparison and verdicts leave pyssp_simulate; measurement and judgment live in pyssp_vv, and permutations may call pyssp_vv.measure locally
