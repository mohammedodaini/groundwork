"""Lightweight run tracing.

Deliberately dependency-free (ADR-007). A hosted tracer like Langfuse is better
for production, but a portfolio project that cannot be run without signing up
for a SaaS account is a portfolio project nobody runs. This records the same
information - spans, durations, errors - into the result and the logs, and
exposes a hook (`on_span`) so a Langfuse exporter can be attached in one place.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    started_at: float
    duration_ms: int = 0
    error: str | None = None


@dataclass
class RunTracer:
    """Collects timing and errors per graph node."""

    spans: list[Span] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    on_span: Callable[[Span], None] | None = None

    @contextmanager
    def span(self, name: str) -> Iterator[Span]:
        span = Span(name=name, started_at=time.perf_counter())
        try:
            yield span
        finally:
            span.duration_ms = int((time.perf_counter() - span.started_at) * 1000)
            self.spans.append(span)
            logger.debug("span", extra={"node": name, "ms": span.duration_ms})
            if self.on_span:
                self.on_span(span)

    def record_error(self, node: str, message: str) -> None:
        self.errors.append((node, message))
        if self.spans and self.spans[-1].name == node:
            self.spans[-1].error = message

    def summary(self) -> dict:
        """Per-node timing, for the /trace endpoint and the UI."""
        by_node: dict[str, dict] = {}
        for s in self.spans:
            entry = by_node.setdefault(s.name, {"calls": 0, "total_ms": 0, "errors": 0})
            entry["calls"] += 1
            entry["total_ms"] += s.duration_ms
            if s.error:
                entry["errors"] += 1
        return {
            "nodes": by_node,
            "total_ms": sum(s.duration_ms for s in self.spans),
            "errors": [{"node": n, "message": m} for n, m in self.errors],
        }
