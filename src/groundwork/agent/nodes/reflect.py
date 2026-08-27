"""Reflection, qualification and synthesis nodes.

`reflect_node` is what makes this agentic rather than a pipeline. It looks at
what has been gathered and decides: is this enough, or do I need another search
round targeting a specific gap? That decision - and the new queries it emits -
is the "I need more information" behaviour the brief asks for.

It is bounded by `max_search_rounds`. An agent that can loop must have a
termination condition that does not depend on the model's judgement alone.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from groundwork.agent.prompts import (
    GAP_ANALYST_SYSTEM,
    INSIGHT_SYSTEM,
    QUALIFIER_SYSTEM,
)
from groundwork.agent.state import ResearchState
from groundwork.domain.enums import EpistemicStatus, QualificationDecision
from groundwork.domain.schemas import Entity, Insight, Qualification
from groundwork.providers.llm import LLMError, LLMProvider

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Reflection / gap analysis
# --------------------------------------------------------------------------


class GapAnalysis(BaseModel):
    model_config = {"extra": "forbid"}

    sufficient: bool
    reason: str = Field(default="", max_length=800)
    new_queries: list[str] = Field(default_factory=list, max_length=3)


def _evidence_summary(state: ResearchState) -> str:
    entities = state.get("entities", [])
    lines = [
        f"Sources fetched: {len(state.get('sources', []))}",
        f"Entities found: {len(entities)}",
        f"Queries already run: {state.get('executed_queries', [])}",
        "",
    ]
    for ent in entities[:10]:
        facts = len(ent.claims_by_status(EpistemicStatus.FACT))
        infs = len(ent.claims_by_status(EpistemicStatus.INFERENCE))
        unk = len(ent.claims_by_status(EpistemicStatus.UNKNOWN))
        lines.append(f"- {ent.name}: {facts} FACT, {infs} INFERENCE, {unk} UNKNOWN")
    return "\n".join(lines)


async def reflect_node(state: ResearchState, *, llm: LLMProvider) -> dict:
    """Decide whether to run another research round."""
    request = state["request"]
    round_index = state.get("round_index", 1)

    # Hard bound first. This check is deliberately before the LLM call so the
    # model cannot talk the system into an extra round.
    if round_index >= request.max_search_rounds:
        return {
            "should_continue_research": False,
            "gap_reason": f"Reached max_search_rounds ({request.max_search_rounds}).",
        }

    entities = state.get("entities", [])
    if len(entities) >= request.max_entities and all(
        e.claims_by_status(EpistemicStatus.FACT) for e in entities
    ):
        return {
            "should_continue_research": False,
            "gap_reason": "Target entity count reached and each has factual support.",
        }

    criteria = "\n".join(f"- {c}" for c in request.criteria) or "(none)"
    user = (
        f"OBJECTIVE:\n{request.objective}\n\n"
        f"CRITERIA:\n{criteria}\n\n"
        f"CURRENT STATE:\n{_evidence_summary(state)}\n\n"
        f"This was round {round_index} of at most {request.max_search_rounds}.\n"
        "Is the evidence sufficient? If not, give up to 3 NEW queries."
    )

    try:
        out = await llm.structured(system=GAP_ANALYST_SYSTEM, user=user, schema=GapAnalysis)
    except Exception as exc:
        # Deliberately broad. This node controls a loop; ANY failure here must
        # stop research rather than leave the routing flag untouched. See the
        # `error_update` comment in agent/graph.py.
        logger.warning("reflect_failed", extra={"error": str(exc)})
        # Fail closed: stop researching rather than loop blindly.
        return {
            "should_continue_research": False,
            "gap_reason": f"Gap analysis unavailable ({exc}); stopping.",
            "warnings": ["Reflection LLM failed; stopped research early."],
        }

    already = set(state.get("executed_queries", []))
    fresh = [q.strip() for q in out.new_queries if q.strip() and q.strip() not in already]

    if out.sufficient or not fresh:
        return {
            "should_continue_research": False,
            "gap_reason": out.reason or "Evidence judged sufficient.",
        }

    logger.info("another_round", extra={"round": round_index + 1, "queries": len(fresh)})
    return {
        "should_continue_research": True,
        "gap_reason": out.reason,
        "queries": fresh,
        "round_index": round_index + 1,
    }


# --------------------------------------------------------------------------
# Qualification
# --------------------------------------------------------------------------


class QualificationOutput(BaseModel):
    model_config = {"extra": "forbid"}

    decision: QualificationDecision
    rationale: str = Field(default="", max_length=2000)
    criteria_met: list[str] = Field(default_factory=list)
    criteria_failed: list[str] = Field(default_factory=list)
    criteria_unknown: list[str] = Field(default_factory=list)


def _render_claims(entity: Entity) -> str:
    if not entity.claims:
        return "(no claims)"
    return "\n".join(
        f"- [{c.status.value}"
        + (f"/{c.verdict.value}" if c.verdict else "")
        + f", conf={c.confidence:.2f}] {c.text}"
        for c in entity.claims
    )


async def qualify_node(state: ResearchState, *, llm: LLMProvider) -> dict:
    """Judge each entity against the user's criteria."""
    request = state["request"]
    entities: list[Entity] = list(state.get("entities", []))

    if not request.criteria:
        # No criteria means nothing to qualify against. Say so rather than
        # inventing a judgement.
        for ent in entities:
            ent.qualification = Qualification(
                decision=QualificationDecision.INSUFFICIENT_EVIDENCE,
                rationale="No qualification criteria were supplied by the user.",
            )
        return {"entities": entities}

    criteria_text = "\n".join(f"- {c}" for c in request.criteria)
    warnings: list[str] = []

    for entity in entities:
        user = (
            f"OBJECTIVE:\n{request.objective}\n\n"
            f"CRITERIA:\n{criteria_text}\n\n"
            f"ENTITY: {entity.name}\n"
            f"VERIFIED CLAIMS:\n{_render_claims(entity)}\n\n"
            "Judge this entity against each criterion."
        )
        try:
            out = await llm.structured(
                system=QUALIFIER_SYSTEM, user=user, schema=QualificationOutput
            )
            entity.qualification = Qualification(
                decision=out.decision,
                rationale=out.rationale,
                criteria_met=out.criteria_met,
                criteria_failed=out.criteria_failed,
                criteria_unknown=out.criteria_unknown,
            )
        except LLMError as exc:
            logger.warning("qualify_failed", extra={"entity": entity.name})
            entity.qualification = Qualification(
                decision=QualificationDecision.INSUFFICIENT_EVIDENCE,
                rationale=f"Qualification could not be completed: {exc}",
            )
            warnings.append(f"Qualification failed for {entity.name}.")

    return {"entities": entities, "warnings": warnings}


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------


class InsightItem(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(min_length=3, max_length=1000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class InsightOutput(BaseModel):
    model_config = {"extra": "forbid"}

    insights: list[InsightItem] = Field(default_factory=list, max_length=4)


async def insight_node(state: ResearchState, *, llm: LLMProvider) -> dict:
    """Produce cross-entity observations. Always INFERENCE, never FACT."""
    entities = state.get("entities", [])
    if len(entities) < 1:
        return {"insights": []}

    rendered = "\n\n".join(f"ENTITY: {e.name}\n{_render_claims(e)}" for e in entities[:10])
    user = (
        f"OBJECTIVE:\n{state['request'].objective}\n\n"
        f"VERIFIED CLAIMS ACROSS ENTITIES:\n{rendered}\n\n"
        "Produce 0-4 cross-entity insights."
    )

    try:
        out = await llm.structured(system=INSIGHT_SYSTEM, user=user, schema=InsightOutput)
    except LLMError as exc:
        logger.warning("insight_failed", extra={"error": str(exc)})
        return {"insights": [], "warnings": ["Insight generation failed."]}

    # Attach every claim id as provenance. Coarse, but honest: we do not
    # pretend to know precisely which claims drove which insight.
    claim_ids = [c.id for e in entities for c in e.claims]
    return {
        "insights": [
            Insight(text=i.text, confidence=i.confidence, supporting_claim_ids=claim_ids[:20])
            for i in out.insights
        ]
    }
