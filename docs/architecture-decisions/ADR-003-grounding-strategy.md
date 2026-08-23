# ADR-003: Verbatim quote verification as the grounding mechanism

**Status:** Accepted · **Date:** 2026-08

## Context
The system's value proposition is that its output can be audited. That requires
a mechanism that distinguishes "the source says this" from "the model believes
this". Self-reported confidence does not do that: a hallucination is often
reported confidently.

## Options considered
1. **Trust the model's own FACT/INFERENCE labels.** Zero cost, zero guarantee.
2. **LLM-as-judge over each claim.** Catches semantic mismatch, but is itself a
   model with an error rate, costs a call per batch, and can be wrong in the
   same direction as the extractor.
3. **Deterministic verbatim matching of quotes against fetched text.** Cannot
   detect a *misleading* quote, but catches *fabricated* quotes with certainty.

## Decision
Layer 3 first, then layer 2 for what layer 3 cannot see.

Concretely: the extractor must supply verbatim quotes for any FACT. We normalise
only for artefacts we ourselves introduce (unicode form, whitespace runs, curly
quotes) and then require an exact substring match. A quote that fails is
discarded and its claim is downgraded FACT → INFERENCE. The LLM critic then runs
on what survives, checking whether the quote actually *supports* the claim.

## Consequences
- `evidence_coverage` and `quote_verification_rate` are deterministic and
  reproducible, so they are honest metrics to report.
- A model that paraphrases instead of quoting is penalised, which is the
  behaviour we want.
- **This does not make the system hallucination-free.** A real quote can be
  cherry-picked, and an INFERENCE can be wrong. The system reduces and *labels*
  unsupported content; it does not eliminate it. Stated plainly in the README.
