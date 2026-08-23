# ADR-002: Where the agent's autonomy begins and ends

**Status:** Accepted · **Date:** 2026-08

## Context
"Agentic" is often taken to mean "the model decides everything". That maximises
demo impressiveness and minimises reliability. We need an explicit boundary.

## Decision
The model has autonomy over **what to look for and what things mean**. It has no
autonomy over **what the system does**.

| Decision | Owner | Why |
| --- | --- | --- |
| Which queries to run | LLM | Genuine language understanding |
| Whether to search again | LLM, hard-bounded by `max_search_rounds` | Genuine judgement, but must terminate |
| What a page claims | LLM | Reading comprehension |
| Whether a quote is real | **Deterministic code** | String matching is exact and free |
| Source quality tier | **Deterministic code** | A stateable rule beats a model call |
| Deduplication | **Deterministic code** | SHA-256 |
| Whether to act on a result | **Human** | Consequential; see ADR-006 |

## Consequences
- Fewer LLM calls per run than a comparable multi-agent design → cheaper, faster,
  less variance.
- The parts that must never be wrong (quote verification, loop termination,
  SSRF policy) cannot be talked out of their behaviour by any prompt.
- Some quality is left on the table: an LLM source-quality judge would beat our
  domain rule on unusual sites. Accepted, and documented as a limitation.
