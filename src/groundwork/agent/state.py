"""The state object threaded through the LangGraph workflow.

A `TypedDict` (not a Pydantic model) because LangGraph merges partial updates
returned by each node, and TypedDict makes that merge explicit and cheap. The
*outputs* are Pydantic (see domain/schemas.py) - validation happens at the
boundaries, not on every internal hop.

Reducers matter here. `operator.add` on the list fields means a node can return
`{"sources": [new_source]}` and LangGraph appends rather than replaces. Getting
this wrong is the single most common LangGraph bug: you silently lose state
because a node returned a shorter list.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from groundwork.domain.schemas import (
    Entity,
    Evidence,
    Insight,
    ResearchRequest,
    RunMetrics,
    Source,
)


class ResearchState(TypedDict, total=False):
    # -- inputs (set once) -------------------------------------------------
    request: ResearchRequest

    # -- planning ----------------------------------------------------------
    plan: list[str]
    queries: Annotated[list[str], operator.add]
    executed_queries: Annotated[list[str], operator.add]

    # -- gathered material -------------------------------------------------
    sources: Annotated[list[Source], operator.add]
    evidence: Annotated[list[Evidence], operator.add]
    # Raw sanitized page text, keyed by source id. Not persisted; used within
    # the run for verbatim verification.
    source_texts: dict[str, str]
    # Source ids the extractor has already seen.
    #
    # This must be tracked EXPLICITLY rather than derived from `evidence`.
    # Deriving it was a real bug: a source whose quotes all failed verbatim
    # verification produces no evidence, so it looked unprocessed and was
    # re-extracted on every subsequent round - burning an LLM call each time and
    # appending duplicate claims, which made one source look like several
    # corroborating ones.
    processed_source_ids: Annotated[list[str], operator.add]

    # -- derived intelligence ----------------------------------------------
    entities: list[Entity]
    insights: list[Insight]

    # -- control flow ------------------------------------------------------
    # The loop counter that stops the agent researching forever. Every agentic
    # system needs one of these; unbounded loops are how demos become bills.
    round_index: int
    should_continue_research: bool
    gap_reason: str

    # -- bookkeeping -------------------------------------------------------
    metrics: RunMetrics
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def initial_state(request: ResearchRequest) -> ResearchState:
    return ResearchState(
        request=request,
        plan=[],
        queries=[],
        executed_queries=[],
        sources=[],
        evidence=[],
        source_texts={},
        processed_source_ids=[],
        entities=[],
        insights=[],
        round_index=0,
        should_continue_research=True,
        gap_reason="",
        metrics=RunMetrics(),
        warnings=[],
        errors=[],
    )
