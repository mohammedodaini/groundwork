"""API and persistence integration tests.

The human-in-the-loop tests matter most here: they assert that a result cannot
move to an approved state without an explicit human decision, and that a second
decision is rejected rather than silently overwriting the first reviewer.
"""

from __future__ import annotations

import time

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


def test_read_endpoints_are_authenticated_when_configured(monkeypatch):
    """Reads are gated too, not just the endpoint that spends money.

    Gating writes alone was a real hole: with API_KEY set, creating a job
    required the key but reading it back did not, so every objective and every
    result was public to anyone who could reach the port. The research payload
    IS the sensitive data here.
    """
    monkeypatch.setenv("API_KEY", "secret-key")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from groundwork.config import get_settings

    get_settings.cache_clear()
    auth = {"X-API-Key": "secret-key"}
    with TestClient(create_app()) as c:
        job_id = c.post(
            "/api/research",
            json={"objective": "A perfectly valid research objective here."},
            headers=auth,
        ).json()["job_id"]

        for path in (
            "/api/research",
            f"/api/research/{job_id}",
            f"/api/research/{job_id}/trace",
        ):
            assert c.get(path).status_code == 401, f"{path} leaked without a key"
            assert c.get(path, headers={"X-API-Key": "wrong"}).status_code == 401

        # The correct key still gets through.
        assert c.get("/api/research", headers=auth).status_code == 200
        assert c.get(f"/api/research/{job_id}", headers=auth).status_code == 200
    get_settings.cache_clear()


def test_reads_stay_open_when_no_key_is_configured(monkeypatch):
    """The frictionless local default must survive the change above."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    from groundwork.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/api/research").status_code == 200
    get_settings.cache_clear()


def test_production_without_api_key_refuses_to_start(monkeypatch):
    """A missing env var must not be the difference between private and public.

    `api_key` defaults to None so `make run` works with no setup. Combined with
    ENVIRONMENT=production that default is an open, paid endpoint, so the app
    refuses to construct its settings at all rather than booting wide open.
    """
    from pydantic import ValidationError

    from groundwork.config import Settings

    monkeypatch.delenv("API_KEY", raising=False)
    # `_env_file=None` keeps the test hermetic: it must not depend on whatever
    # the developer happens to have in their local .env.
    with pytest.raises(ValidationError, match="requires API_KEY"):
        Settings(_env_file=None, environment="production")

    ok = Settings(_env_file=None, environment="production", api_key="a-real-key")
    assert ok.api_key is not None
    assert ok.api_key.get_secret_value() == "a-real-key"

    # Local and test environments keep the frictionless default.
    assert Settings(_env_file=None, environment="local").api_key is None


def _wait_for_job(client, job_id: str, timeout: float = 10.0) -> dict:
    """Poll until the job leaves PENDING/RUNNING. Jobs run as background tasks."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/research/{job_id}").json()
        if body["status"] not in ("PENDING", "RUNNING"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


def test_offline_mode_replays_the_demo_instead_of_an_empty_report(client):
    """`make run` on a fresh clone must show a real-looking run.

    The API used to build a FakeLLM with no script: the planner failed, nothing
    was searched, and the UI showed an empty report waiting for approval.
    """
    res = client.post(
        "/api/research",
        json={"objective": "Find technical wholesalers in Gelderland processing orders manually."},
    )
    body = _wait_for_job(client, res.json()["job_id"])

    result = body["result"]
    claims = [c for entity in result["entities"] for c in entity["claims"]]
    assert result["entities"], "the demo scenario finds entities"
    assert any(c["status"] == "FACT" for c in claims)
    assert result["warnings"][0].startswith("Offline demo mode")


# --------------------------------------------------------------------------
# Runs a previous process never finished
# --------------------------------------------------------------------------


async def test_unfinished_jobs_are_failed_when_the_server_restarts(repo, settings):
    pending = await repo.create_job(
        ResearchRequest(objective="A research objective, never started.")
    )
    running = await repo.create_job(ResearchRequest(objective="A research objective, mid-flight."))
    await repo.mark_running(running)
    done_req = ResearchRequest(
        objective="A research objective that finished.", require_approval=False
    )
    done = await repo.create_job(done_req)
    await repo.save_result(done, await _make_result(settings, done_req))

    assert await repo.fail_interrupted_jobs() == 2
    for job_id in (pending, running):
        row = await repo.get_job(job_id)
        assert row.status == JobStatus.FAILED.value
        assert row.error.startswith("Interrupted")
    assert (await repo.get_job(done)).status == JobStatus.COMPLETED.value


def _app_env(monkeypatch, db_url: str) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("SEARCH_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    from groundwork.config import get_settings

    get_settings.cache_clear()


def test_restart_fails_runs_a_crashed_process_left_running(tmp_path, monkeypatch):
    """A crash leaves a RUNNING row behind; the next startup must close it out."""
    import asyncio

    db_url = f"sqlite+aiosqlite:///{tmp_path}/crash.db"

    async def crash_mid_run() -> str:
        db = Database(db_url)
        await db.create_all()
        repo = ResearchRepository(db)
        job_id = await repo.create_job(
            ResearchRequest(objective="A research run that never ended.")
        )
        await repo.mark_running(job_id)
        await db.dispose()
        return job_id

    job_id = asyncio.run(crash_mid_run())
    _app_env(monkeypatch, db_url)
    with TestClient(create_app()) as c:
        body = c.get(f"/api/research/{job_id}").json()

    assert body["status"] == JobStatus.FAILED.value
    assert body["error"].startswith("Interrupted")


def test_shutdown_records_cancelled_runs_as_interrupted(tmp_path, monkeypatch):
    """A run cancelled at shutdown raises CancelledError, which `except Exception`
    does not catch, so it used to stay RUNNING in the database."""
    import asyncio

    import groundwork.main as main_module

    class _StuckEngine:
        async def run(self, request):
            await asyncio.Event().wait()

    db_url = f"sqlite+aiosqlite:///{tmp_path}/shutdown.db"
    _app_env(monkeypatch, db_url)
    monkeypatch.setattr(main_module, "SHUTDOWN_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(main_module, "build_engine", lambda settings: _StuckEngine())

    with TestClient(create_app()) as c:
        job_id = c.post(
            "/api/research", json={"objective": "A research run that will not finish."}
        ).json()["job_id"]
        deadline = time.monotonic() + 5
        while c.get(f"/api/research/{job_id}").json()["status"] != "RUNNING":
            assert time.monotonic() < deadline, "job never started"
            time.sleep(0.02)

    async def read_back():
        db = Database(db_url)
        row = await ResearchRepository(db).get_job(job_id)
        await db.dispose()
        return row

    row = asyncio.run(read_back())  # read directly: no startup cleanup involved
    assert row.status == JobStatus.FAILED.value
    assert row.error.startswith("Interrupted")


def test_trace_survives_a_restart(tmp_path, monkeypatch):
    """Traces lived in a dict on app.state: never evicted, and gone after a restart."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/trace.db"
    _app_env(monkeypatch, db_url)
    with TestClient(create_app()) as c:
        job_id = c.post(
            "/api/research",
            json={"objective": "Find technical wholesalers in Gelderland processing orders."},
        ).json()["job_id"]
        _wait_for_job(c, job_id)
        assert c.get(f"/api/research/{job_id}/trace").status_code == 200

    _app_env(monkeypatch, db_url)
    with TestClient(create_app()) as c:
        trace = c.get(f"/api/research/{job_id}/trace")

    assert trace.status_code == 200
    assert trace.json(), "the trace should carry per-node timings"
