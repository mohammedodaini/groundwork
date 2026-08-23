"""Shared fixtures.

The whole suite runs offline: FakeLLM and FakeSearch replace the network, and
SQLite replaces Postgres. That is a deliberate property - CI must not need API
keys, and a recruiter cloning this repo must be able to run `pytest` and see it
pass in seconds.
"""

from __future__ import annotations

import json

import pytest

from groundwork.config import Settings
from groundwork.domain.schemas import ResearchRequest
from groundwork.providers.llm import FakeLLM
from groundwork.providers.search import ContentFetcher, FakeSearch, SearchHit


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        llm_provider="fake",
        search_provider="fake",
        database_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        rate_limit_per_minute=0,  # disabled in tests
    )


PAGE_TEXT = (
    "Acme Technical Wholesale B.V. is a technical wholesaler based in Arnhem, "
    "Gelderland. The company was founded in 1998 and employs 45 staff. "
    "Acme supplies industrial fasteners and hydraulic components to "
    "manufacturers across the Netherlands. Orders are currently processed "
    "manually by the sales desk."
)


def plan_response(queries: list[str] | None = None) -> str:
    return json.dumps(
        {
            "reasoning": "Break the objective into region and industry queries.",
            "steps": ["Find wholesalers", "Check region", "Check size"],
            "queries": queries or ["technical wholesaler Gelderland", "groothandel Arnhem"],
        }
    )


def extraction_response(
    *,
    entity: str = "Acme Technical Wholesale B.V.",
    quote: str = "The company was founded in 1998 and employs 45 staff.",
    status: str = "FACT",
    relevant: bool = True,
) -> str:
    return json.dumps(
        {
            "entity_name": entity,
            "relevant": relevant,
            "claims": [
                {
                    "text": "Acme was founded in 1998 and has 45 employees.",
                    "status": status,
                    "quotes": [quote],
                    "confidence": 0.9,
                }
            ],
            "injection_attempt_noted": False,
        }
    )


def critic_response(verdict: str = "SUPPORTED", count: int = 1) -> str:
    return json.dumps(
        {
            "verdicts": [
                {
                    "claim_index": i,
                    "verdict": verdict,
                    "note": "Checked against quote.",
                    "adjusted_confidence": 0.8,
                }
                for i in range(count)
            ]
        }
    )


def gap_response(sufficient: bool = True, queries: list[str] | None = None) -> str:
    return json.dumps(
        {
            "sufficient": sufficient,
            "reason": "Enough evidence." if sufficient else "Need employee counts.",
            "new_queries": queries or [],
        }
    )


def qualification_response(decision: str = "QUALIFIED") -> str:
    return json.dumps(
        {
            "decision": decision,
            "rationale": "Meets region and industry criteria.",
            "criteria_met": ["Located in Gelderland"],
            "criteria_failed": [],
            "criteria_unknown": [],
        }
    )


def insight_response(n: int = 1) -> str:
    return json.dumps(
        {
            "insights": [
                {"text": f"Observation {i}: manual order processing is common.", "confidence": 0.6}
                for i in range(n)
            ]
        }
    )


def happy_path_responses() -> list[str]:
    """One full successful run: plan, extract, critic, gap, qualify, insight."""
    return [
        plan_response(),
        extraction_response(),
        critic_response(),
        gap_response(sufficient=True),
        qualification_response(),
        insight_response(),
    ]


@pytest.fixture
def fake_llm(settings: Settings) -> FakeLLM:
    return FakeLLM(settings, responses=happy_path_responses())


@pytest.fixture
def fake_search(settings: Settings) -> FakeSearch:
    return FakeSearch(
        settings,
        default=[SearchHit(url="https://acme-wholesale.nl/about", title="About Acme")],
    )


class StubFetcher(ContentFetcher):
    """Returns canned page text without touching the network."""

    def __init__(self, settings: Settings, pages: dict[str, str] | None = None) -> None:
        super().__init__(settings)
        self.pages = pages or {"https://acme-wholesale.nl/about": PAGE_TEXT}

    async def fetch(self, url: str, *, entity_domain: str | None = None):
        from groundwork.domain.schemas import Source
        from groundwork.providers.search import FetchResult
        from groundwork.security.sanitize import detect_injection, neutralise

        if url not in self.pages:
            return None
        text = neutralise(self.pages[url])
        flags = detect_injection(text)
        self.fetch_count += 1
        return FetchResult(
            source=Source(
                url=url,
                title="stub",
                content_sha256=Source.hash_content(text),
                char_count=len(text),
                injection_flags=flags,
            ),
            text=text,
        )


@pytest.fixture
def stub_fetcher(settings: Settings) -> StubFetcher:
    return StubFetcher(settings)


@pytest.fixture
def request_obj() -> ResearchRequest:
    return ResearchRequest(
        objective="Find technical wholesalers in Gelderland that process orders manually.",
        criteria=["Located in Gelderland", "Technical wholesaler"],
        max_entities=3,
        max_search_rounds=2,
    )
