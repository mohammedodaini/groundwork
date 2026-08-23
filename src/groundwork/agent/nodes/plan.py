"""Planning node: objective -> concrete search queries.

This node is genuinely LLM-shaped work: turning "find Dutch technical
wholesalers in Gelderland that could benefit from AI automation" into queries a
search engine will answer well requires language understanding. Compare with
source tiering, which is a rule - we do not use an LLM there. Deciding which is
which is the core judgement of this project (ADR-003).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from groundwork.agent.prompts import PLANNER_SYSTEM
from groundwork.agent.state import ResearchState
from groundwork.providers.llm import LLMError, LLMProvider

logger = logging.getLogger(__name__)


class PlanOutput(BaseModel):
    """Structured output contract for the planner."""

    model_config = {"extra": "forbid"}

    reasoning: str = Field(default="", max_length=1500)
    steps: list[str] = Field(default_factory=list, max_length=8)
    queries: list[str] = Field(min_length=1, max_length=8)


def _fallback_queries(objective: str, criteria: list[str]) -> list[str]:
    """Deterministic degradation when the planner LLM is unavailable.

    A research tool that returns nothing because one LLM call failed is a bad
    tool. These queries are worse than the model's, but they are not nothing.
    """
    base = objective.strip()
    queries = [base]
    if criteria:
        queries.append(f"{base} {criteria[0]}")
    queries.append(f"{base} official site")
    return queries[:3]


async def plan_node(state: ResearchState, *, llm: LLMProvider) -> dict:
    """Produce the research plan and the first round of queries."""
    request = state["request"]
    criteria_text = "\n".join(f"- {c}" for c in request.criteria) or "(none specified)"

    user = (
        f"RESEARCH OBJECTIVE:\n{request.objective}\n\n"
        f"QUALIFICATION CRITERIA:\n{criteria_text}\n\n"
        f"Target: up to {request.max_entities} entities.\n"
        "Produce a short plan and the search queries to start with."
    )

    try:
        out = await llm.structured(system=PLANNER_SYSTEM, user=user, schema=PlanOutput)
        queries = [q.strip() for q in out.queries if q.strip()]
        steps = out.steps or [f"Search for: {q}" for q in queries]
        logger.info("plan_created", extra={"query_count": len(queries)})
        return {"plan": steps, "queries": queries, "round_index": 1}
    except LLMError as exc:
        logger.error("planner_failed", extra={"error": str(exc)})
        fallback = _fallback_queries(request.objective, request.criteria)
        return {
            "plan": ["Fallback plan: direct search on the objective."],
            "queries": fallback,
            "round_index": 1,
            "warnings": [f"Planner LLM failed ({exc}); used fallback queries."],
        }
