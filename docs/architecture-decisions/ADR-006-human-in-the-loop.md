# ADR-006: No external actions, enforced by omission

**Status:** Accepted · **Date:** 2026-08

## Context
The original brief imagined outreach ("companies that could benefit from
automation" → email them). An agent that reads attacker-controlled web pages and
can send email is a prompt-injection weapon: a crafted page could plausibly
cause outbound messages.

## Decision
Two independent guarantees:

1. **Capability isolation.** The extraction and critic LLM calls have **no tools
   bound**. A perfectly successful injection has nothing to call.
2. **Enforcement by omission.** The application implements no outbound action at
   all. `POST /decision` records a human's approval and writes it to the
   database. That is the entire effect.

Runs default to `require_approval=true` and terminate in `AWAITING_APPROVAL`.
A second decision on an already-decided job returns 409 rather than silently
overwriting the recorded reviewer.

## Consequences
- The "human in the loop" is real rather than a UI affordance over an
  auto-executing pipeline.
- If outbound actions are added later, they must go behind the approval gate and
  the threat model must be revisited. Noted in Future Work.
- The strongest security property in this project comes from a feature we chose
  *not* to build.
