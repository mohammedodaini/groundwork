"""Pydantic schemas: the contract between the LLM, the graph, the DB, and the API.

Design rule followed throughout: **every claim must be traceable to evidence,
and every piece of evidence must be traceable to a source.** The types below
enforce that structurally rather than by convention, so a claim with a dangling
evidence reference is a validation error, not a silent bug.

    Source (a URL we fetched)
      -> Evidence (a verbatim span from that source)
           -> Claim (an assertion, citing >= 0 evidence)
                -> Entity (a thing we researched, holding claims)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from groundwork.domain.enums import (
    EpistemicStatus,
    QualificationDecision,
    SourceTier,
    VerificationVerdict,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(BaseModel):
    """Shared config. `extra="forbid"` is deliberate.

    LLMs love to invent extra keys. Forbidding them turns a silent schema drift
    into a loud ValidationError that our retry logic can feed back to the model.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# --------------------------------------------------------------------------
# Sources and evidence
# --------------------------------------------------------------------------


class Source(Base):
    """A document we actually retrieved. Never a URL the LLM merely recalled."""

    id: UUID = Field(default_factory=uuid4)
    url: HttpUrl
    title: str = ""
    domain: str = ""
    tier: SourceTier = SourceTier.UNKNOWN
    fetched_at: datetime = Field(default_factory=_utcnow)
    content_sha256: str = ""
    char_count: int = 0
    # Set by the sanitizer when the page contained injection-like patterns.
    injection_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_domain(self) -> Source:
        if not self.domain:
            host = self.url.host or ""
            object.__setattr__(self, "domain", host.lower().removeprefix("www."))
        return self

    @staticmethod
    def hash_content(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


class Evidence(Base):
    """A verbatim span lifted from a Source.

    `quote` must appear in the fetched source text. The extraction node checks
    this (see `agent/nodes/extract.py`); spans that fail the check are dropped.
    That single check is our strongest anti-fabrication mechanism, because it is
    deterministic — it does not ask an LLM whether the LLM was honest.
    """

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    quote: Annotated[str, Field(min_length=1, max_length=2000)]
    # Character offset in the sanitized source text, when locatable.
    start_char: int | None = None
    verbatim_verified: bool = False

    @field_validator("quote")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


class Claim(Base):
    """An assertion the system is willing to stand behind, with its epistemics."""

    id: UUID = Field(default_factory=uuid4)
    text: Annotated[str, Field(min_length=3, max_length=1000)]
    status: EpistemicStatus
    evidence_ids: list[UUID] = Field(default_factory=list)
    # 0..1. Calibration is not claimed; this is a coarse self-report that the
    # critic can override. See ADR-005 for why we keep it and distrust it.
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    verdict: VerificationVerdict | None = None
    critic_note: str = ""

    @model_validator(mode="after")
    def _fact_requires_evidence(self) -> Claim:
        """A FACT with no evidence is a contradiction in terms.

        We downgrade rather than raise: the critic stage is the right place to
        surface this to the user, and raising here would abort an otherwise
        useful run over one bad claim.
        """
        if self.status is EpistemicStatus.FACT and not self.evidence_ids:
            object.__setattr__(self, "status", EpistemicStatus.INFERENCE)
            object.__setattr__(
                self,
                "critic_note",
                (self.critic_note + " | auto-downgraded: FACT without evidence").strip(" |"),
            )
        return self


# --------------------------------------------------------------------------
# Entities, qualification, insights
# --------------------------------------------------------------------------


class Qualification(Base):
    """Does this entity meet the user's criteria, and why."""

    decision: QualificationDecision
    rationale: Annotated[str, Field(max_length=2000)] = ""
    criteria_met: list[str] = Field(default_factory=list)
    criteria_failed: list[str] = Field(default_factory=list)
    criteria_unknown: list[str] = Field(default_factory=list)


class Entity(Base):
    """A researched thing (usually an organisation)."""

    id: UUID = Field(default_factory=uuid4)
    name: Annotated[str, Field(min_length=1, max_length=300)]
    canonical_url: HttpUrl | None = None
    claims: list[Claim] = Field(default_factory=list)
    qualification: Qualification | None = None

    def claims_by_status(self, status: EpistemicStatus) -> list[Claim]:
        return [c for c in self.claims if c.status is status]


class Insight(Base):
    """A cross-entity observation. Always an INFERENCE by construction."""

    id: UUID = Field(default_factory=uuid4)
    text: Annotated[str, Field(min_length=3, max_length=1000)]
    supporting_claim_ids: list[UUID] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


# --------------------------------------------------------------------------
# Requests and results
# --------------------------------------------------------------------------


class ResearchRequest(Base):
    """What the user asked for."""

    objective: Annotated[str, Field(min_length=10, max_length=2000)]
    criteria: list[str] = Field(default_factory=list)
    max_entities: Annotated[int, Field(ge=1, le=25)] = 5
    max_search_rounds: Annotated[int, Field(ge=1, le=6)] = 3
    # Whether results require human sign-off before being marked COMPLETED.
    require_approval: bool = True


class RunMetrics(Base):
    """Measured, never estimated by an LLM."""

    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    latency_ms: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    search_calls: int = 0
    pages_fetched: int = 0
    estimated_cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class ResearchResult(Base):
    """The full auditable output of one research run."""

    request: ResearchRequest
    plan: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _referential_integrity(self) -> ResearchResult:
        """Guarantee no dangling references before this leaves the system.

        This is the invariant the whole UI depends on: if a claim cites
        evidence, that evidence exists and points at a real source.
        """
        source_ids = {s.id for s in self.sources}
        evidence_ids = {e.id for e in self.evidence}

        for ev in self.evidence:
            if ev.source_id not in source_ids:
                raise ValueError(f"Evidence {ev.id} references unknown source {ev.source_id}")

        for ent in self.entities:
            for claim in ent.claims:
                unknown = set(claim.evidence_ids) - evidence_ids
                if unknown:
                    raise ValueError(f"Claim {claim.id} references unknown evidence {unknown}")
        return self

    # -- convenience accessors used by the API, UI and evaluator --------------

    @property
    def all_claims(self) -> list[Claim]:
        return [c for e in self.entities for c in e.claims]

    def unsupported_claim_rate(self) -> float:
        """Share of claims the critic could not support. Lower is better."""
        claims = self.all_claims
        if not claims:
            return 0.0
        bad = sum(
            1
            for c in claims
            if c.verdict
            in {
                VerificationVerdict.UNSUPPORTED,
                VerificationVerdict.CONTRADICTED,
                VerificationVerdict.OVERSTATED,
            }
        )
        return bad / len(claims)

    def evidence_coverage(self) -> float:
        """Share of FACT claims that carry at least one verbatim-verified span."""
        facts = [c for c in self.all_claims if c.status is EpistemicStatus.FACT]
        if not facts:
            return 0.0
        verified = {e.id for e in self.evidence if e.verbatim_verified}
        return sum(1 for c in facts if set(c.evidence_ids) & verified) / len(facts)
