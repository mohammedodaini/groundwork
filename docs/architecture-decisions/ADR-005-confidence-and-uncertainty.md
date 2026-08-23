# ADR-005: Keep self-reported confidence, but distrust it

**Status:** Accepted · **Date:** 2026-08

## Context
LLM self-reported confidence is poorly calibrated. Displaying "0.92" implies a
precision we cannot support.

## Decision
Keep the number, but never let it stand alone:
- It is always shown next to an epistemic status and a critic verdict.
- The critic can and does overwrite it; structural failures clamp it to ≤ 0.3.
- No downstream logic thresholds on it. Nothing is filtered by confidence alone.
- The README states it is uncalibrated.

## Alternatives rejected
- **Removing it.** Loses a useful relative signal within a single run.
- **Calibrating it.** Would need a labelled dataset we do not have. Claiming
  calibration without that dataset would be the dishonest option.
