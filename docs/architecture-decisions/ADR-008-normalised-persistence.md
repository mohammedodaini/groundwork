# ADR-008: Normalise the evidence graph; keep a JSON read model alongside

**Status:** Accepted · **Date:** 2026-08

## Context
Simplest option: one `result_json` column per job. Fast to write, and adequate
for rendering a single result.

## Decision
Normalised tables for `sources`, `entities`, `claims`, `evidence` — plus the
denormalised `result_json` snapshot.

## Rationale
The premise of the project is auditability. The question that proves it is
*"show me every unsupported claim across all runs, with its evidence and source
tier"*. Against a JSON column that is painful; against `claims.verdict`
(indexed) it is one query, which the evaluation harness uses.

The JSON snapshot stays because the API's common path is "render one complete
historical result", and doing that from normalised tables is an N+1 waiting to
happen. Classic write-model / read-model split.

## Consequences
- Two representations to keep consistent; both are written in one transaction in
  `save_result`.
- Schema is created with `Base.metadata.create_all`. That is not a migration
  strategy — a production system needs Alembic. Listed under Limitations rather
  than glossed over.
