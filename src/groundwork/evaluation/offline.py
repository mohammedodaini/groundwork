"""Offline providers that actually exercise the graph.

`--provider fake` used to build a bare `FakeLLM`, which returns `"{}"` for every
call. Every structured-output node then failed validation three times and gave
up, so the harness reported "8/8 completed without error" while producing zero
claims. A green run that measures nothing is worse than a red one.

These providers replace that. The LLM is still fake, but it returns
schema-valid responses, and its extraction quotes real sentences from the page
it was given -- so the verbatim-verification path runs for real. Claims are
emitted in pairs on purpose:

  * one whose quote IS present in the source -> should stay FACT
  * one whose quote is NOT present          -> should be downgraded

That makes the offline run a regression test for the mechanism the whole
project rests on, instead of a smoke test that proves the process starts.

Only the network is faked. Search returns fixture hits, and fetches are served
through an httpx MockTransport, so the real fetch path still runs -- SSRF
policy, size caps, HTML-to-text, and injection sanitisation included.
"""

from __future__ import annotations

import json
import re

import httpx

from groundwork.config import Settings
from groundwork.providers.llm import FakeLLM, LLMProvider
from groundwork.providers.search import (
    ContentFetcher,
    FakeSearch,
    SearchHit,
    SearchProvider,
)

# --------------------------------------------------------------------------
# Fixture corpus
# --------------------------------------------------------------------------

# Each page carries one distinctive sentence. The extractor quotes it verbatim,
# which is what lets quote verification succeed. Keep them plain: they have to
# survive HTML-to-text conversion and sanitisation unchanged.
_VERIFIABLE = {
    "https://example.com/report-a": (
        "The organisation published its annual figures on 12 March 2026 "
        "and reported a headcount of 240 people."
    ),
    "https://example.com/report-b": (
        "The programme runs for three years and is taught entirely in English."
    ),
    "https://example.com/report-c": (
        "The regulation entered into force in August 2024 and applies in stages."
    ),
}

# Asserted as FACT but present in no source. The pipeline must downgrade it.
FABRICATED_QUOTE = "This sentence appears in no source document whatsoever."

_PAGE_TEMPLATE = """<!doctype html>
<html><head><title>{title}</title></head>
<body>
<h1>{title}</h1>
<p>{sentence}</p>
<p>This paragraph is filler so the page is not trivially short. It exists to
give the chunker and the sanitiser something realistic to work on.</p>
</body></html>
"""


def _handler(request: httpx.Request) -> httpx.Response:
    """Serve the fixture corpus. Unknown URLs 404 rather than hitting network."""
    url = str(request.url)
    sentence = _VERIFIABLE.get(url)
    if sentence is None:
        return httpx.Response(404, text="not found")
    title = url.rsplit("/", 1)[-1].replace("-", " ").title()
    return httpx.Response(
        200,
        text=_PAGE_TEMPLATE.format(title=title, sentence=sentence),
        headers={"content-type": "text/html; charset=utf-8"},
    )


def build_offline_search(settings: Settings) -> SearchProvider:
    """Search that always returns the fixture corpus."""
    hits = [
        SearchHit(url=url, title=url.rsplit("/", 1)[-1], snippet=sentence[:120])
        for url, sentence in _VERIFIABLE.items()
    ]
    return FakeSearch(settings, default=hits)


def build_offline_fetcher(settings: Settings) -> ContentFetcher:
    """Real fetch pipeline, fixture bytes."""
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        follow_redirects=False,
        timeout=5.0,
    )
    return ContentFetcher(settings, client=client)


# --------------------------------------------------------------------------
# Schema-aware fake LLM
# --------------------------------------------------------------------------


_KNOWN_SCHEMAS = frozenset(
    {
        "PlanOutput",
        "ExtractionOutput",
        "CriticOutput",
        "GapAnalysis",
        "InsightOutput",
        "QualificationOutput",
    }
)


def _schema_name(user: str) -> str:
    """`structured()` embeds the JSON Schema in the prompt; read its title.

    Every *property* carries a "title" too, and those come first in the
    serialised schema, so match against the known model names rather than
    taking the first hit.
    """
    for candidate in re.findall(r'"title":\s*"(\w+)"', user):
        if candidate in _KNOWN_SCHEMAS:
            return candidate
    return ""


def _quotable_sentence(user: str) -> str | None:
    """Return whichever fixture sentence is present in this prompt."""
    for sentence in _VERIFIABLE.values():
        if sentence in user:
            return sentence
    return None


def offline_router(system: str, user: str) -> str:
    """Return a schema-valid response for whichever node is calling."""
    name = _schema_name(user)

    if name == "PlanOutput":
        return json.dumps(
            {
                "reasoning": "Offline fixture plan.",
                "steps": ["Search the fixture corpus", "Extract and verify quotes"],
                "queries": ["fixture corpus overview", "fixture corpus details"],
            }
        )

    if name == "ExtractionOutput":
        verifiable = _quotable_sentence(user)
        claims: list[dict] = []
        if verifiable:
            claims.append(
                {
                    "text": "The source states a specific, checkable detail.",
                    "status": "FACT",
                    "quotes": [verifiable],
                    "confidence": 0.9,
                }
            )
        # Always emitted. Its quote is in no source, so verification must
        # downgrade it -- that is the behaviour under test.
        claims.append(
            {
                "text": "An assertion the source does not actually support.",
                "status": "FACT",
                "quotes": [FABRICATED_QUOTE],
                "confidence": 0.9,
            }
        )
        claims.append(
            {
                "text": "The organisation is plausibly mid-sized given the figures.",
                "status": "INFERENCE",
                "quotes": [],
                "confidence": 0.5,
            }
        )
        return json.dumps(
            {
                "entity_name": "Fixture Entity",
                "relevant": True,
                "claims": claims,
                "injection_attempt_noted": False,
            }
        )

    if name == "CriticOutput":
        # Defer to the deterministic structural audit rather than inventing
        # verdicts: that audit is the part worth exercising offline.
        return json.dumps({"verdicts": []})

    if name == "GapAnalysis":
        return json.dumps(
            {
                "sufficient": True,
                "reason": "Fixture corpus is fully consumed.",
                "new_queries": [],
            }
        )

    if name == "QualificationOutput":
        # INSUFFICIENT_EVIDENCE is the honest verdict against a fixture corpus
        # that was never written to satisfy any real criterion.
        return json.dumps(
            {
                "decision": "INSUFFICIENT_EVIDENCE",
                "rationale": "The fixture corpus does not speak to these criteria.",
                "criteria_met": [],
                "criteria_failed": [],
                "criteria_unknown": [],
            }
        )

    if name == "InsightOutput":
        return json.dumps(
            {
                "insights": [
                    {
                        "text": "Both fixture sources describe the same entity.",
                        "confidence": 0.4,
                    }
                ]
            }
        )

    # Unknown schema: an empty object is still the honest answer, and the
    # repair loop will surface it as a real failure rather than hiding it.
    return "{}"


def build_offline_llm(settings: Settings) -> LLMProvider:
    """A fake LLM that answers every node with valid, quotable output."""
    return FakeLLM(settings, router=offline_router)
