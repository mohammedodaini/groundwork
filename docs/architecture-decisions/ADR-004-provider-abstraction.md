# ADR-004: Abstract LLM and search behind interfaces

**Status:** Accepted · **Date:** 2026-08

## Context
Calling `anthropic.messages.create()` directly from node code is the shortest
path to a working demo and the longest path to a testable system.

## Decision
`LLMProvider` and `SearchProvider` interfaces, with `FakeLLM` / `FakeSearch`
implementations used by the entire test suite.

## Consequences
- **The whole suite runs offline with no API key.** This is the property that
  makes the repo credible: `git clone && pytest` passes in seconds, for anyone.
- Retries, timeouts, token accounting and structured-output repair live in one
  place instead of being copy-pasted per call site.
- Switching provider is one config value; the OpenAI implementation exists to
  prove the abstraction is real rather than aspirational.
- Cost: one more indirection layer, and `FakeLLM` mocks the transport, not the
  intelligence. Tests verify how the graph handles good, malformed and failing
  responses — not whether a real model would produce them.
