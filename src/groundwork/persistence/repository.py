"""Database session management and the research repository.

The repository is the only place that knows about SQLAlchemy. Everything above
it (API, graph, evaluation) speaks in Pydantic domain objects. That boundary is
what lets the test-suite run entirely against in-memory SQLite while production
runs on Postgres, with no code changes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import selectinload

from groundwork.domain.enums import JobStatus
from groundwork.domain.schemas import ResearchRequest, ResearchResult
from groundwork.persistence.models import (
    Base,
    ClaimRow,
    EntityRow,
    EvidenceRow,
    JobRow,
    SourceRow,
)

logger = logging.getLogger(__name__)


class Database:
    """Owns the engine and session factory."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self._engine = create_async_engine(url, echo=echo, future=True)
        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    async def create_all(self) -> None:
        """Create tables.

        Fine for a portfolio project and for tests. A production deployment
        would use Alembic migrations; noted as a known limitation in the README
        rather than pretending `create_all` is a migration strategy.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


class ResearchRepository:
    """CRUD for research jobs and their evidence graph."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def create_job(self, request: ResearchRequest) -> str:
        async with self.db.session() as s:
            row = JobRow(
                objective=request.objective,
                criteria=list(request.criteria),
                status=JobStatus.PENDING.value,
            )
            s.add(row)
            await s.flush()
            return row.id

    async def mark_running(self, job_id: str) -> None:
        await self._set_status(job_id, JobStatus.RUNNING)

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self.db.session() as s:
            row = await s.get(JobRow, job_id)
            if row:
                row.status = JobStatus.FAILED.value
                row.error = error[:4000]

    async def _set_status(self, job_id: str, status: JobStatus) -> None:
        async with self.db.session() as s:
            row = await s.get(JobRow, job_id)
            if row:
                row.status = status.value

    async def save_result(self, job_id: str, result: ResearchResult) -> None:
        """Persist the full evidence graph plus the denormalised snapshot."""
        async with self.db.session() as s:
            job = await s.get(JobRow, job_id)
            if job is None:
                raise KeyError(f"Unknown job {job_id}")

            job.status = (
                JobStatus.AWAITING_APPROVAL.value
                if result.request.require_approval
                else JobStatus.COMPLETED.value
            )
            job.result_json = result.model_dump(mode="json")
            job.latency_ms = result.metrics.latency_ms
            job.llm_calls = result.metrics.llm_calls
            job.prompt_tokens = result.metrics.prompt_tokens
            job.completion_tokens = result.metrics.completion_tokens
            job.estimated_cost_usd = result.metrics.estimated_cost_usd

            for src in result.sources:
                s.add(
                    SourceRow(
                        id=str(src.id),
                        job_id=job_id,
                        url=str(src.url),
                        title=src.title,
                        domain=src.domain,
                        tier=src.tier.value,
                        content_sha256=src.content_sha256,
                        char_count=src.char_count,
                        injection_flags=list(src.injection_flags),
                        fetched_at=src.fetched_at,
                    )
                )

            evidence_by_id = {str(e.id): e for e in result.evidence}
            attached: set[str] = set()

            for ent in result.entities:
                s.add(
                    EntityRow(
                        id=str(ent.id),
                        job_id=job_id,
                        name=ent.name,
                        canonical_url=str(ent.canonical_url) if ent.canonical_url else None,
                        qualification_decision=(
                            ent.qualification.decision.value if ent.qualification else None
                        ),
                        qualification_rationale=(
                            ent.qualification.rationale if ent.qualification else None
                        ),
                    )
                )
                for claim in ent.claims:
                    s.add(
                        ClaimRow(
                            id=str(claim.id),
                            entity_id=str(ent.id),
                            text=claim.text,
                            status=claim.status.value,
                            verdict=claim.verdict.value if claim.verdict else None,
                            confidence=claim.confidence,
                            critic_note=claim.critic_note,
                        )
                    )
                    for eid in claim.evidence_ids:
                        ev = evidence_by_id.get(str(eid))
                        if ev is None:
                            continue
                        attached.add(str(ev.id))
                        s.add(
                            EvidenceRow(
                                id=str(ev.id),
                                claim_id=str(claim.id),
                                source_id=str(ev.source_id),
                                quote=ev.quote,
                                start_char=ev.start_char,
                                verbatim_verified=ev.verbatim_verified,
                            )
                        )

            # Orphan evidence (extracted but not cited) is still recorded - it
            # is part of the audit trail of what the system looked at.
            for ev in result.evidence:
                if str(ev.id) not in attached:
                    s.add(
                        EvidenceRow(
                            id=str(ev.id),
                            claim_id=None,
                            source_id=str(ev.source_id),
                            quote=ev.quote,
                            start_char=ev.start_char,
                            verbatim_verified=ev.verbatim_verified,
                        )
                    )

    async def decide(
        self, job_id: str, *, approved: bool, reviewer: str, note: str = ""
    ) -> bool:
        """Record a human approval decision. Returns False if not pending."""
        from datetime import datetime

        async with self.db.session() as s:
            row = await s.get(JobRow, job_id)
            if row is None or row.status != JobStatus.AWAITING_APPROVAL.value:
                return False
            row.status = (
                JobStatus.APPROVED.value if approved else JobStatus.REJECTED.value
            )
            row.approved_by = reviewer[:200]
            row.approval_note = note[:4000]
            row.decided_at = datetime.now(UTC)
            return True

    async def get_job(self, job_id: str) -> JobRow | None:
        async with self.db.session() as s:
            return await s.get(JobRow, job_id)

    async def list_jobs(self, *, limit: int = 50) -> list[JobRow]:
        async with self.db.session() as s:
            res = await s.execute(
                select(JobRow).order_by(JobRow.created_at.desc()).limit(limit)
            )
            return list(res.scalars().all())

    async def unsupported_claims(self, *, limit: int = 100) -> list[ClaimRow]:
        """The query that justifies the normalised schema.

        Used by the evaluation harness to inspect failure modes across runs.
        """
        async with self.db.session() as s:
            res = await s.execute(
                select(ClaimRow)
                .options(selectinload(ClaimRow.evidence))
                .where(ClaimRow.verdict.in_(["UNSUPPORTED", "CONTRADICTED", "OVERSTATED"]))
                .limit(limit)
            )
            return list(res.scalars().all())
