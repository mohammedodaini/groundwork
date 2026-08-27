"""Verification / critic node.

Two layers, in this order:

1. DETERMINISTIC CHECKS (`structural_audit`) - things we can decide without a
   model: a FACT with no verified evidence, evidence from a page flagged for
   injection, a claim whose only source is SECONDARY tier. These are free and
   never wrong.
2. LLM CRITIC - semantic checks a rule cannot do: does the quote actually
   *establish* the claim, or merely mention the topic? Is the claim stronger
   than the evidence ("is expanding" vs "announced plans to explore expanding")?

Running the cheap deterministic layer first means the LLM critic sees fewer
claims and we spend fewer tokens. It also means the system degrades sensibly:
if the critic LLM is down, the structural audit still runs.

HONEST LIMITATION: this reduces unsupported claims, it does not eliminate them.
The critic is the same class of system as the extractor and shares some of its
blind spots. We report the residual rate rather than claiming zero.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from groundwork.agent.prompts import CRITIC_SYSTEM
from groundwork.agent.state import ResearchState
from groundwork.domain.enums import EpistemicStatus, SourceTier, VerificationVerdict
from groundwork.domain.schemas import Claim, Entity, Evidence, Source
from groundwork.providers.llm import LLMError, LLMProvider

logger = logging.getLogger(__name__)

MAX_CLAIMS_PER_CRITIC_CALL = 12


class ClaimVerdict(BaseModel):
    model_config = {"extra": "forbid"}

    claim_index: int = Field(ge=0)
    verdict: VerificationVerdict
    note: str = Field(default="", max_length=500)
    adjusted_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CriticOutput(BaseModel):
    model_config = {"extra": "forbid"}

    verdicts: list[ClaimVerdict] = Field(default_factory=list)


def structural_audit(
    claim: Claim,
    evidence_by_id: dict[str, Evidence],
    sources_by_id: dict[str, Source],
) -> tuple[VerificationVerdict | None, str]:
    """Deterministic checks. Returns (verdict, note) or (None, "") to defer."""
    if claim.status is EpistemicStatus.UNKNOWN:
        return VerificationVerdict.SUPPORTED, "UNKNOWN is self-consistent."

    linked = [evidence_by_id[str(i)] for i in claim.evidence_ids if str(i) in evidence_by_id]

    if claim.status is EpistemicStatus.FACT:
        if not linked:
            return VerificationVerdict.UNSUPPORTED, "FACT with no evidence attached."
        if not any(e.verbatim_verified for e in linked):
            return (
                VerificationVerdict.UNSUPPORTED,
                "FACT whose quotes failed verbatim verification.",
            )

    # Evidence originating only from injection-flagged pages is untrustworthy.
    if linked:
        # Bind to a new name so the type narrows: reassigning `srcs` keeps the
        # Optional in the element type and hides genuine None bugs.
        resolved = [
            src for src in (sources_by_id.get(str(e.source_id)) for e in linked) if src is not None
        ]
        if resolved and all(s.injection_flags for s in resolved):
            return (
                VerificationVerdict.UNSUPPORTED,
                "All supporting sources contained prompt-injection patterns.",
            )
        if (
            resolved
            and all(s.tier is SourceTier.SECONDARY for s in resolved)
            and claim.confidence > 0.8
        ):
            return (
                VerificationVerdict.OVERSTATED,
                "High confidence asserted from secondary sources only.",
            )

    return None, ""


def _format_claims_for_critic(
    claims: list[Claim],
    evidence_by_id: dict[str, Evidence],
    sources_by_id: dict[str, Source],
) -> str:
    lines: list[str] = []
    for idx, claim in enumerate(claims):
        lines.append(f"[{idx}] CLAIM ({claim.status.value}): {claim.text}")
        if not claim.evidence_ids:
            lines.append("    EVIDENCE: (none)")
        for eid in claim.evidence_ids:
            ev = evidence_by_id.get(str(eid))
            if ev is None:
                continue
            src = sources_by_id.get(str(ev.source_id))
            domain = src.domain if src else "unknown"
            tier = src.tier.value if src else "UNKNOWN"
            verified = "verified" if ev.verbatim_verified else "UNVERIFIED"
            lines.append(f'    EVIDENCE ({domain}, {tier}, {verified}): "{ev.quote[:400]}"')
        lines.append("")
    return "\n".join(lines)


async def critic_node(state: ResearchState, *, llm: LLMProvider) -> dict:
    """Assign a verdict to every claim, revising status and confidence."""
    entities: list[Entity] = list(state.get("entities", []))
    if not entities:
        return {"warnings": ["Critic called with no entities."]}

    evidence_by_id = {str(e.id): e for e in state.get("evidence", [])}
    sources_by_id = {str(s.id): s for s in state.get("sources", [])}

    warnings: list[str] = []
    needs_llm: list[Claim] = []

    # -- layer 1: deterministic ------------------------------------------
    for entity in entities:
        for claim in entity.claims:
            # Skip claims already audited in an earlier round.
            #
            # The critic runs once per research round, but claims accumulate
            # across rounds. Without this guard a round-2 run re-audits every
            # round-1 claim: duplicated critic notes, duplicated token spend,
            # and confidence repeatedly overwritten. Found by running the demo,
            # which surfaced notes reading "Quote states the claim. | Quote
            # states the claim."
            if claim.verdict is not None:
                continue
            verdict, note = structural_audit(claim, evidence_by_id, sources_by_id)
            if verdict is not None:
                claim.verdict = verdict
                claim.critic_note = (claim.critic_note + " | " + note).strip(" |")
                if verdict is not VerificationVerdict.SUPPORTED:
                    claim.confidence = min(claim.confidence, 0.3)
            else:
                needs_llm.append(claim)

    # -- layer 2: LLM semantic critic -------------------------------------
    for start in range(0, len(needs_llm), MAX_CLAIMS_PER_CRITIC_CALL):
        batch = needs_llm[start : start + MAX_CLAIMS_PER_CRITIC_CALL]
        rendered = _format_claims_for_critic(batch, evidence_by_id, sources_by_id)
        user = (
            f"RESEARCH OBJECTIVE:\n{state['request'].objective}\n\n"
            f"CLAIMS TO AUDIT:\n{rendered}\n"
            "Return a verdict for every claim index shown above."
        )
        try:
            out = await llm.structured(system=CRITIC_SYSTEM, user=user, schema=CriticOutput)
        except LLMError as exc:
            logger.warning("critic_failed", extra={"error": str(exc)})
            warnings.append(f"Critic LLM failed on a batch ({exc}); claims left unverified.")
            for claim in batch:
                claim.critic_note = (claim.critic_note + " | critic unavailable").strip(" |")
            continue

        seen: set[int] = set()
        for v in out.verdicts:
            if v.claim_index >= len(batch):
                continue
            seen.add(v.claim_index)
            claim = batch[v.claim_index]
            claim.verdict = v.verdict
            claim.critic_note = (claim.critic_note + " | " + v.note).strip(" |")
            claim.confidence = v.adjusted_confidence
            if (
                v.verdict
                in {
                    VerificationVerdict.UNSUPPORTED,
                    VerificationVerdict.CONTRADICTED,
                }
                and claim.status is EpistemicStatus.FACT
            ):
                claim.status = EpistemicStatus.INFERENCE

        # A critic that skips claims is a silent failure; make it loud.
        for idx, claim in enumerate(batch):
            if idx not in seen:
                claim.critic_note = (claim.critic_note + " | critic returned no verdict").strip(
                    " |"
                )
                warnings.append(f"Critic returned no verdict for claim: {claim.text[:60]!r}")

    total = sum(len(e.claims) for e in entities)
    unsupported = sum(
        1
        for e in entities
        for c in e.claims
        if c.verdict
        in {
            VerificationVerdict.UNSUPPORTED,
            VerificationVerdict.CONTRADICTED,
            VerificationVerdict.OVERSTATED,
        }
    )
    logger.info("critic_complete", extra={"claims": total, "unsupported": unsupported})

    return {"entities": entities, "warnings": warnings}
