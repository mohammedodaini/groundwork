"""API and persistence integration tests.

The human-in-the-loop tests matter most here: they assert that a result cannot
move to an approved state without an explicit human decision, and that a second
decision is rejected rather than silently overwriting the first reviewer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from groundwork.domain.enums import JobStatus
from groundwork.domain.schemas import ResearchRequest
from groundwork.main import create_app
from groundwork.persistence.repository import Database, ResearchRepository

# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


@pytest.fixture
async def repo():
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    yield ResearchRepository(db)
    await db.dispose()


async def _make_result(settings, request_obj):
    from tests.conftest import StubFetcher, happy_path_responses

    from groundwork.agent.graph import ResearchEngine
    from groundwork.providers.llm import FakeLLM
    from groundwork.providers.search import FakeSearch, SearchHit

    engine = ResearchEngine(
        settings=settings,
        llm=FakeLLM(settings, responses=happy_path_responses()),
        search=FakeSearch(settings, default=[SearchHit(url="https://acme-wholesale.nl/about")]),
        fetcher=StubFetcher(settings),
    )
    return await engine.run(request_obj)


async def test_result_survives_and_is_retrievable(repo, settings, request_obj):
    """Milestone 6's definition of done."""
    job_id = await repo.create_job(request_obj)
    result = await _make_result(settings, request_obj)
    await repo.save_result(job_id, result)

    row = await repo.get_job(job_id)
    assert row is not None
    assert row.status == JobStatus.AWAITING_APPROVAL.value
    assert row.result_json["entities"][0]["name"] == "Acme Technical Wholesale B.V."
    assert row.llm_calls > 0


async def test_evidence_graph_is_queryable(repo, settings, request_obj):
    """Justifies the normalised schema over a JSON blob."""
    from sqlalchemy import func, select

    from groundwork.persistence.models import ClaimRow, EvidenceRow

    job_id = await repo.create_job(request_obj)
    await repo.save_result(job_id, await _make_result(settings, request_obj))

    async with repo.db.session() as s:
        claims = (await s.execute(select(func.count()).select_from(ClaimRow))).scalar()
        verified = (
            await s.execute(
                select(func.count())
                .select_from(EvidenceRow)
                .where(EvidenceRow.verbatim_verified.is_(True))
            )
        ).scalar()
    assert claims >= 1
    assert verified >= 1


async def test_approval_flow(repo, settings, request_obj):
    job_id = await repo.create_job(request_obj)
    await repo.save_result(job_id, await _make_result(settings, request_obj))

    assert await repo.decide(job_id, approved=True, reviewer="mo", note="looks right")
    row = await repo.get_job(job_id)
    assert row.status == JobStatus.APPROVED.value
    assert row.approved_by == "mo"


async def test_second_decision_is_refused(repo, settings, request_obj):
    """A double-click must not overwrite the recorded reviewer."""
    job_id = await repo.create_job(request_obj)
    await repo.save_result(job_id, await _make_result(settings, request_obj))
    assert await repo.decide(job_id, approved=True, reviewer="first")
    assert not await repo.decide(job_id, approved=False, reviewer="second")
    row = await repo.get_job(job_id)
    assert row.approved_by == "first"


async def test_cannot_approve_unknown_job(repo):
    assert not await repo.decide("does-not-exist", approved=True, reviewer="x")


async def test_run_without_approval_completes_directly(repo, settings):
    req = ResearchRequest(
        objective="A research objective that is long enough.", require_approval=False
    )
    job_id = await repo.create_job(req)
    await repo.save_result(job_id, await _make_result(settings, req))
    row = await repo.get_job(job_id)
    assert row.status == JobStatus.COMPLETED.value


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    from groundwork.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_touches_database(client):
    assert client.get("/ready").json()["status"] == "ready"


def test_create_and_poll_job(client):
    res = client.post(
        "/api/research",
        json={
            "objective": "Find technical wholesalers in Gelderland processing orders manually.",
            "criteria": ["Located in Gelderland"],
            "max_entities": 2,
            "max_search_rounds": 1,
        },
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    got = client.get(f"/api/research/{job_id}")
    assert got.status_code == 200
    assert got.json()["objective"].startswith("Find technical wholesalers")


def test_unknown_job_returns_404(client):
    assert client.get("/api/research/nope").status_code == 404


def test_invalid_request_is_rejected(client):
    """Objective below min_length must 422, not reach the LLM."""
    res = client.post("/api/research", json={"objective": "short"})
    assert res.status_code == 422


def test_extra_fields_rejected(client):
    res = client.post(
        "/api/research",
        json={
            "objective": "A perfectly valid research objective here.",
            "not_a_real_field": True,
        },
    )
    assert res.status_code == 422


def test_decision_on_non_pending_job_conflicts(client):
    res = client.post(
        "/api/research",
        json={"objective": "A perfectly valid research objective here."},
    )
    job_id = res.json()["job_id"]
    # Immediately after creation the job is PENDING, not AWAITING_APPROVAL.
    decision = client.post(
        f"/api/research/{job_id}/decision",
        json={"approved": True, "reviewer": "mo"},
    )
    assert decision.status_code == 409


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from groundwork.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        body = {"objective": "A perfectly valid research objective here."}
        assert c.post("/api/research", json=body).status_code == 401
        assert c.post("/api/research", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
        assert (
            c.post("/api/research", json=body, headers={"X-API-Key": "secret-key"}).status_code
            == 202
        )
    get_settings.cache_clear()


def test_health_is_not_authenticated(monkeypatch):
    """Ops endpoints must stay reachable when auth is on."""
    monkeypatch.setenv("API_KEY", "secret-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from groundwork.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200
    get_settings.cache_clear()
