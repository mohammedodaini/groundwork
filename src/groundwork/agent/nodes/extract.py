"""Extraction node: page text -> entities, claims, and verbatim evidence.

THE IMPORTANT PART OF THIS FILE is `verify_quote`. The LLM is asked to supply
verbatim quotes; we then *check* them against the page text with plain string
matching. Quotes that do not appear are discarded and their claims downgraded.

This is deterministic verification of a probabilistic system. It cannot be
talked out of its verdict, it costs nothing, and it is the mechanism that makes
`evidence_coverage` a number worth reporting. Asking a second LLM "did the first
LLM quote accurately?" would be slower, cost more, and be wrong sometimes.

Note the LLM here has NO tools bound. Even a fully successful prompt injection
in page content cannot trigger an external action from this node.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from uuid import UUID

from pydantic import BaseModel, Field

from groundwork.agent.prompts import EXTRACTOR_SYSTEM, extraction_user_prompt
from groundwork.agent.state import ResearchState
from groundwork.domain.enums import EpistemicStatus
from groundwork.domain.schemas import Claim, Entity, Evidence, Source
from groundwork.providers.llm import LLMError, LLMProvider
from groundwork.security.sanitize import wrap_untrusted

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# LLM output contract
# --------------------------------------------------------------------------


class RawClaim(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(min_length=3, max_length=1000)
    status: EpistemicStatus
    quotes: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractionOutput(BaseModel):
    model_config = {"extra": "forbid"}

    entity_name: str = Field(default="", max_length=300)
    relevant: bool = True
    claims: list[RawClaim] = Field(default_factory=list, max_length=15)
    # Set by the model if the page tried to instruct it. Useful telemetry:
    # it tells us whether the model *noticed* the injection we detected by regex.
    injection_attempt_noted: bool = False


# --------------------------------------------------------------------------
# Verbatim verification
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Normalise for comparison without destroying the quote.

    We forgive exactly three things, because they are artefacts of HTML-to-text
    rather than fabrication: unicode form, whitespace runs, and curly quotes.
    We forgive nothing else - not word changes, not truncation, not paraphrase.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(quote: str, page_text: str) -> tuple[bool, int | None]:
    """Is `quote` genuinely present in `page_text`?

    Returns (verified, start_offset). The offset is into the normalised text and
    is used only for UI highlighting, so approximate is fine.
    """
    if not quote.strip():
        return False, None
    norm_quote = _normalise(quote)
    norm_page = _normalise(page_text)
    if len(norm_quote) < 8:
        # Too short to be meaningful evidence; trivially matches common words.
        return False, None
    idx = norm_page.find(norm_quote)
    return (idx != -1), (idx if idx != -1 else None)


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------


async def extract_from_source(
    *,
    llm: LLMProvider,
    objective: str,
    source: Source,
    page_text: str,
) -> tuple[str, list[Claim], list[Evidence], list[str]]:
    """Extract from one page. Returns (entity_name, claims, evidence, warnings)."""
    warnings: list[str] = []

    wrapped = wrap_untrusted(page_text)
    user = extraction_user_prompt(objective, str(source.url), wrapped)

    try:
        out = await llm.structured(
            system=EXTRACTOR_SYSTEM, user=user, schema=ExtractionOutput
        )
    except LLMError as exc:
        logger.warning("extraction_failed", extra={"url": str(source.url), "error": str(exc)})
        return "", [], [], [f"Extraction failed for {source.domain}: {exc}"]

    if not out.relevant or not out.claims:
        return out.entity_name, [], [], warnings

    if out.injection_attempt_noted:
        warnings.append(
            f"Model reported an instruction-like passage on {source.domain}."
        )

    claims: list[Claim] = []
    evidence: list[Evidence] = []

    for raw in out.claims:
        verified_ids: list[UUID] = []

        for quote in raw.quotes:
            ok, offset = verify_quote(quote, page_text)
            ev = Evidence(
                source_id=source.id,
                quote=quote[:2000],
                start_char=offset,
                verbatim_verified=ok,
            )
            if ok:
                evidence.append(ev)
                verified_ids.append(ev.id)
            else:
                # This is the money line: a fabricated quote is caught here,
                # deterministically, and never reaches the user as a FACT.
                warnings.append(
                    f"Discarded unverifiable quote on {source.domain}: "
                    f"{quote[:80]!r}"
                )

        status = raw.status
        note = ""
        if status is EpistemicStatus.FACT and not verified_ids:
            # Claimed as fact but no quote survived verification.
            status = EpistemicStatus.INFERENCE
            note = "Downgraded FACT->INFERENCE: no verbatim quote could be verified."

        claims.append(
            Claim(
                text=raw.text,
                status=status,
                evidence_ids=verified_ids,
                confidence=raw.confidence,
                critic_note=note,
            )
        )

    return out.entity_name, claims, evidence, warnings


def _merge_entities(existing: list[Entity], name: str, claims: list[Claim]) -> list[Entity]:
    """Attach claims to an entity, creating it if new. Case-insensitive match."""
    if not name.strip():
        name = "Unnamed entity"
    key = name.strip().lower()
    for ent in existing:
        if ent.name.strip().lower() == key:
            ent.claims.extend(claims)
            return existing
    existing.append(Entity(name=name.strip(), claims=claims))
    return existing


async def extract_node(state: ResearchState, *, llm: LLMProvider) -> dict:
    """Extract structured intelligence from every not-yet-processed source."""
    objective = state["request"].objective
    texts = state.get("source_texts", {})

    # Explicitly tracked, NOT derived from `evidence`. A source whose quotes all
    # failed verification yields zero evidence but has still been processed;
    # deriving this set caused repeated re-extraction and duplicate claims.
    processed = set(state.get("processed_source_ids", []))
    pending = [
        s
        for s in state.get("sources", [])
        if str(s.id) in texts and str(s.id) not in processed
    ]

    if not pending:
        return {"warnings": ["Extract called with no unprocessed sources."]}

    entities = list(state.get("entities", []))
    all_evidence: list[Evidence] = []
    all_warnings: list[str] = []

    for source in pending:
        name, claims, evidence, warnings = await extract_from_source(
            llm=llm,
            objective=objective,
            source=source,
            page_text=texts[str(source.id)],
        )
        all_evidence.extend(evidence)
        all_warnings.extend(warnings)
        if claims:
            entities = _merge_entities(entities, name, claims)

    logger.info(
        "extract_complete",
        extra={"sources": len(pending), "entities": len(entities)},
    )

    return {
        "entities": entities,
        "evidence": all_evidence,
        "warnings": all_warnings,
        # Mark every source as processed whether or not it yielded evidence.
        "processed_source_ids": [str(s.id) for s in pending],
    }
