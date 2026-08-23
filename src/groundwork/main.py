"""FastAPI application.

Human-in-the-loop is enforced *here*, at the API boundary, not inside the graph.
A research run always stops at AWAITING_APPROVAL; there is no code path in this
application that performs an outbound action (email, CRM write) on a result that
has not been explicitly approved by a human. The safest way to guarantee that is
to not implement the outbound action at all, which is exactly what we do - the
approval endpoint records a decision and nothing else. See ADR-006.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from groundwork.agent.graph import ResearchEngine
from groundwork.config import Settings, get_settings
from groundwork.domain.enums import JobStatus
from groundwork.domain.schemas import ResearchRequest
from groundwork.logging_conf import configure_logging
from groundwork.persistence.repository import Database, ResearchRepository
from groundwork.providers.llm import build_llm
from groundwork.providers.search import ContentFetcher, build_search

logger = logging.getLogger(__name__)

STATIC_DIR = "static"


# --------------------------------------------------------------------------
# Lifespan and dependency wiring
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    db = Database(settings.database_url)
    await db.create_all()
    app.state.settings = settings
    app.state.db = db
    app.state.repo = ResearchRepository(db)
    logger.info(
        "startup",
        extra={"llm": settings.llm_provider, "search": settings.search_provider},
    )
    try:
        yield
    finally:
        await db.dispose()


def get_repo(request: Request) -> ResearchRepository:
    return request.app.state.repo


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def build_engine(settings: Settings) -> ResearchEngine:
    """Fresh providers per run so token counters are per-run, not global."""
    return ResearchEngine(
        settings=settings,
        llm=build_llm(settings),
        search=build_search(settings),
        fetcher=ContentFetcher(settings),
    )


# --------------------------------------------------------------------------
# Auth and rate limiting
# --------------------------------------------------------------------------


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """No-op when no key is configured, so local dev stays frictionless."""
    settings: Settings = request.app.state.settings
    if settings.api_key is None:
        return
    import secrets as _secrets

    expected = settings.api_key.get_secret_value()
    # Constant-time compare: a naive `!=` leaks key material via timing.
    if not x_api_key or not _secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


_hits: dict[str, deque[float]] = defaultdict(deque)


async def rate_limit(request: Request) -> None:
    """In-process sliding window.

    Honest about its limits: this is per-process and resets on restart. Behind
    more than one worker you want Redis. It is here because an endpoint that
    triggers paid LLM calls should never be unbounded, and a simple correct
    limiter beats a sophisticated absent one.
    """
    settings: Settings = request.app.state.settings
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded; retry shortly."
        )
    window.append(now)


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------


class JobCreated(BaseModel):
    job_id: str
    status: str


class ApprovalRequest(BaseModel):
    model_config = {"extra": "forbid"}

    approved: bool
    reviewer: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=4000)


class JobSummary(BaseModel):
    job_id: str
    objective: str
    status: str
    created_at: str
    entity_count: int = 0
    claim_count: int = 0
    unsupported_rate: float = 0.0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Groundwork Research Agent",
        version="0.1.0",
        description=(
            "An evidence-grounded research agent. Every claim is labelled "
            "FACT / INFERENCE / UNKNOWN and traced to verbatim evidence."
        ),
        lifespan=lifespan,
    )

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ---- health -------------------------------------------------------

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready(repo: Annotated[ResearchRepository, Depends(get_repo)]) -> dict:
        """Readiness actually touches the DB. A health check that cannot fail
        is not a health check."""
        try:
            await repo.list_jobs(limit=1)
        except Exception as exc:
            raise HTTPException(503, f"Database unavailable: {exc}") from exc
        return {"status": "ready"}

    # ---- research -----------------------------------------------------

    @app.post(
        "/api/research",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key), Depends(rate_limit)],
        tags=["research"],
    )
    async def start_research(
        request: ResearchRequest,
        raw: Request,
        repo: Annotated[ResearchRepository, Depends(get_repo)],
        settings: Annotated[Settings, Depends(get_app_settings)],
    ) -> JobCreated:
        """Start a run. Returns immediately; poll GET /api/research/{id}."""
        job_id = await repo.create_job(request)

        async def _run() -> None:
            await repo.mark_running(job_id)
            try:
                engine = build_engine(settings)
                result = await engine.run(request)
                await repo.save_result(job_id, result)
                raw.app.state.traces[job_id] = engine.tracer.summary()
            except Exception as exc:
                logger.exception("job_failed", extra={"job_id": job_id})
                await repo.mark_failed(job_id, str(exc))

        # Fire-and-forget is acceptable for a single-node portfolio app; a
        # production deployment would use a real queue (documented in README).
        task = asyncio.create_task(_run())
        raw.app.state.tasks.add(task)
        task.add_done_callback(raw.app.state.tasks.discard)

        return JobCreated(job_id=job_id, status=JobStatus.PENDING.value)

    @app.get("/api/research/{job_id}", tags=["research"])
    async def get_research(
        job_id: str, repo: Annotated[ResearchRepository, Depends(get_repo)]
    ) -> JSONResponse:
        row = await repo.get_job(job_id)
        if row is None:
            raise HTTPException(404, "Job not found")
        return JSONResponse(
            {
                "job_id": row.id,
                "objective": row.objective,
                "status": row.status,
                "created_at": row.created_at.isoformat(),
                "error": row.error,
                "approved_by": row.approved_by,
                "approval_note": row.approval_note,
                "result": row.result_json,
            }
        )

    @app.get("/api/research", response_model=list[JobSummary], tags=["research"])
    async def list_research(
        repo: Annotated[ResearchRepository, Depends(get_repo)], limit: int = 50
    ) -> list[JobSummary]:
        rows = await repo.list_jobs(limit=min(limit, 200))
        out: list[JobSummary] = []
        for r in rows:
            res = r.result_json or {}
            entities = res.get("entities", [])
            claims = [c for e in entities for c in e.get("claims", [])]
            bad = sum(
                1
                for c in claims
                if c.get("verdict") in {"UNSUPPORTED", "CONTRADICTED", "OVERSTATED"}
            )
            out.append(
                JobSummary(
                    job_id=r.id,
                    objective=r.objective,
                    status=r.status,
                    created_at=r.created_at.isoformat(),
                    entity_count=len(entities),
                    claim_count=len(claims),
                    unsupported_rate=round(bad / len(claims), 3) if claims else 0.0,
                    latency_ms=r.latency_ms,
                    estimated_cost_usd=r.estimated_cost_usd,
                )
            )
        return out

    @app.get("/api/research/{job_id}/trace", tags=["observability"])
    async def get_trace(job_id: str, raw: Request) -> dict:
        trace = raw.app.state.traces.get(job_id)
        if trace is None:
            raise HTTPException(404, "No trace for this job")
        return trace

    # ---- human in the loop ---------------------------------------------

    @app.post(
        "/api/research/{job_id}/decision",
        dependencies=[Depends(require_api_key)],
        tags=["human-in-the-loop"],
    )
    async def decide(
        job_id: str,
        body: ApprovalRequest,
        repo: Annotated[ResearchRepository, Depends(get_repo)],
    ) -> dict:
        """Approve or reject a completed run.

        Only valid from AWAITING_APPROVAL. Deliberately not idempotent-silent:
        a second decision returns 409 so a double-click cannot overwrite the
        recorded reviewer.
        """
        ok = await repo.decide(
            job_id, approved=body.approved, reviewer=body.reviewer, note=body.note
        )
        if not ok:
            raise HTTPException(409, "Job is not awaiting approval (or does not exist)")
        return {
            "job_id": job_id,
            "status": JobStatus.APPROVED.value if body.approved else JobStatus.REJECTED.value,
        }

    # ---- static UI ------------------------------------------------------

    import os

    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # In-memory side tables. Not durable by design; traces are debugging aids.
    app.state.tasks = set()
    app.state.traces = {}
    return app


app = create_app()
