# ADR-001: Use LangGraph for orchestration

**Status:** Accepted · **Date:** 2026-08

## Context
The workflow is not a straight line. After criticising extracted claims the
system must decide whether to search again, and that decision depends on
accumulated state. We need: durable state across steps, a conditional edge, a
bounded loop, and per-step observability.

## Options considered
1. **Plain `async` functions in a `while` loop.** Fewest dependencies. But we
   would hand-write state merging, loop bounds, per-node error containment and
   tracing — and we would get the merge semantics subtly wrong first time.
2. **LangChain AgentExecutor / ReAct.** The model chooses tools freely. Wrong
   shape here: our control flow is *known*. Plan → gather → extract → critic is
   not something a model should re-derive on every run, and letting it do so
   adds cost, latency and variance for no benefit.
3. **LangGraph.** Explicit graph, typed state with reducers, conditional edges,
   built-in recursion limit, and a checkpointer story if we later need
   pause/resume mid-graph.

## Decision
LangGraph.

## Consequences
- The one genuinely agentic decision (`reflect` → loop or stop) is a single,
  reviewable conditional edge rather than being buried in prompt text.
- State reducers (`operator.add`) must be declared correctly or nodes silently
  clobber earlier results. This bit us and is now covered by tests.
- We accept a heavier dependency than option 1 in exchange for not writing a
  worse version of it ourselves.

## What we did NOT do
We did not build a multi-agent system with "researcher", "analyst" and "writer"
agents chatting to each other. That architecture is popular in demos and would
have looked more impressive in a diagram, but here it would be three LLM calls
where one deterministic function plus one LLM call does the job more reliably
and more cheaply. See ADR-003.
