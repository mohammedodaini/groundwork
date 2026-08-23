# ADR-007: Built-in tracing rather than a hosted tracer

**Status:** Accepted · **Date:** 2026-08

## Context
LangSmith/Langfuse give richer traces than anything we would write. But a
portfolio repo that cannot start without a SaaS signup is a repo nobody runs.

## Decision
A small `RunTracer` recording per-node spans, durations and errors, exposed at
`GET /api/research/{id}/trace`. It has an `on_span` hook so a Langfuse exporter
can be attached in exactly one place.

## Consequences
- Zero-config debugging: you can answer "which node failed and how long did it
  take" out of the box.
- Traces are in-process and lost on restart. Acceptable for a debugging aid;
  the durable record is the database. Stated as a limitation.
- Token/cost accounting lives in the provider layer, so it is captured whether
  or not a tracer is attached.
