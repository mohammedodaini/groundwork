"""SQLAlchemy ORM models.

Design choice (ADR-008): claims, evidence and sources are stored in *normalised*
tables rather than as one JSON blob per job. A blob would have been faster to
write, but the whole premise of this system is that a claim is traceable to
evidence - and you cannot ask "show me every unsupported claim across all runs"
of a JSON column without pain. The normalised schema makes the audit story real.

`result_json` is kept alongside as a denormalised snapshot so the API can return
a complete historical result without N+1 queries. Classic read-model trade-off.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid_str() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "research_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    objective: Mapped[str] = mapped_column(Text)
    criteria: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Denormalised read model. Nullable until the run finishes.
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Human-in-the-loop
    approved_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    approval_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Metrics, flattened for cheap aggregate queries across runs
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    sources: Mapped[list[SourceRow]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    entities: Mapped[list[EntityRow]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class SourceRow(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("research_jobs.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, default="")
    domain: Mapped[str] = mapped_column(String(255), index=True, default="")
    tier: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    content_sha256: Mapped[str] = mapped_column(String(64), index=True, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    injection_flags: Mapped[list] = mapped_column(JSON, default=list)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job: Mapped[JobRow] = relationship(back_populates="sources")
    evidence: Mapped[list[EvidenceRow]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class EntityRow(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("research_jobs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(300), index=True)
    canonical_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    qualification_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    qualification_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped[JobRow] = relationship(back_populates="entities")
    claims: Mapped[list[ClaimRow]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )


class ClaimRow(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    # Indexed because "find every UNSUPPORTED claim" is the query that makes
    # this schema worth having.
    status: Mapped[str] = mapped_column(String(16), index=True)
    verdict: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    critic_note: Mapped[str] = mapped_column(Text, default="")

    entity: Mapped[EntityRow] = relationship(back_populates="claims")
    evidence: Mapped[list[EvidenceRow]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )


class EvidenceRow(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    claim_id: Mapped[str | None] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), index=True, nullable=True
    )
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    quote: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verbatim_verified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    claim: Mapped[ClaimRow | None] = relationship(back_populates="evidence")
    source: Mapped[SourceRow] = relationship(back_populates="evidence")
