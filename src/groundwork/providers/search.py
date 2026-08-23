"""Web search and page fetching.

Two responsibilities, deliberately separated:

* `SearchProvider` - given a query, return candidate URLs. Swappable.
* `ContentFetcher`  - given a URL, return sanitized text. This is where the
  SSRF guard and the injection sanitiser are enforced, so *every* path into
  the system that touches the open internet goes through one function.

Source tiering is deterministic (a rule you can read), not an LLM judgement.
Asking a model "is this a reputable source?" adds cost, latency and variance to
a decision that a domain-suffix rule makes acceptably well.
"""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, HttpUrl

from groundwork.config import Settings
from groundwork.domain.enums import SourceTier
from groundwork.domain.schemas import Source
from groundwork.security.sanitize import detect_injection, neutralise
from groundwork.security.ssrf import (
    MAX_REDIRECTS,
    MAX_RESPONSE_BYTES,
    UnsafeURLError,
    assert_url_is_safe,
)

logger = logging.getLogger(__name__)


class SearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""


class SearchError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Source tiering
# --------------------------------------------------------------------------

_REPUTABLE_SUFFIXES = (".gov", ".edu", ".ac.uk", ".europa.eu", ".overheid.nl")
_REPUTABLE_DOMAINS = frozenset(
    {
        "reuters.com", "apnews.com", "bbc.co.uk", "ft.com", "economist.com",
        "nature.com", "science.org", "arxiv.org", "nos.nl", "nrc.nl",
        "volkskrant.nl", "fd.nl", "kvk.nl", "cbs.nl",
    }
)
_SECONDARY_DOMAINS = frozenset(
    {
        "wikipedia.org", "medium.com", "reddit.com", "quora.com",
        "linkedin.com", "facebook.com", "x.com", "twitter.com",
        "crunchbase.com", "glassdoor.com", "yelp.com",
    }
)


def classify_source(url: str, *, entity_domain: str | None = None) -> SourceTier:
    """Tier a URL by deterministic rule.

    `entity_domain` lets us mark an organisation's own website as PRIMARY for
    claims about itself - the strongest evidence for "what does this company
    say it does", and the weakest for "is this company any good".
    """
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if not host:
        return SourceTier.UNKNOWN
    if entity_domain and host.endswith(entity_domain.lower().removeprefix("www.")):
        return SourceTier.PRIMARY
    if host.endswith(_REPUTABLE_SUFFIXES) or host in _REPUTABLE_DOMAINS:
        return SourceTier.REPUTABLE
    if any(host.endswith(d) for d in _SECONDARY_DOMAINS):
        return SourceTier.SECONDARY
    return SourceTier.UNKNOWN


# --------------------------------------------------------------------------
# Search providers
# --------------------------------------------------------------------------


class SearchProvider:
    """Interface for web search."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.call_count = 0

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        raise NotImplementedError


class FakeSearch(SearchProvider):
    """Deterministic search for tests and the offline demo."""

    def __init__(
        self,
        settings: Settings,
        *,
        results: dict[str, list[SearchHit]] | None = None,
        default: list[SearchHit] | None = None,
        fail_times: int = 0,
    ) -> None:
        super().__init__(settings)
        self._results = results or {}
        self._default = default or []
        self._fail_times = fail_times
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        self.queries.append(query)
        self.call_count += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise SearchError("Simulated search failure")
        for key, hits in self._results.items():
            if key.lower() in query.lower():
                return hits[:limit]
        return self._default[:limit]


class TavilySearch(SearchProvider):  # pragma: no cover - network dependent
    """Tavily search API. Chosen because it returns clean extracted content."""

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if settings.tavily_api_key is None:
            raise SearchError("TAVILY_API_KEY is required for search_provider=tavily")
        self._key = settings.tavily_api_key.get_secret_value()

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        self.call_count += 1
        payload = {
            "api_key": self._key,
            "query": query,
            "max_results": limit,
            "search_depth": self.settings.tavily_search_depth,
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.fetch_timeout_seconds) as c:
                res = await c.post(self.ENDPOINT, json=payload)
                res.raise_for_status()
                data = res.json()
        except Exception as exc:
            raise SearchError(f"Tavily request failed: {exc}") from exc

        return [
            SearchHit(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", "")[:500],
            )
            for r in data.get("results", [])
            if r.get("url")
        ]


class HttpJsonSearch(SearchProvider):  # pragma: no cover - network dependent
    """Shared plumbing for search APIs that are a GET returning JSON.

    Subclasses supply the request and the parse, so adding a provider is two
    small methods rather than another copy of the client, timeout and
    error-mapping code. Errors are normalised to `SearchError` because
    `gather_node` treats one failed query as survivable, not fatal.
    """

    def _request(self, query: str, limit: int) -> tuple[str, dict, dict]:
        """Return (url, params, headers)."""
        raise NotImplementedError

    def _parse(self, data: dict) -> list[SearchHit]:
        raise NotImplementedError

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        self.call_count += 1
        url, params, headers = self._request(query, limit)
        try:
            async with httpx.AsyncClient(timeout=self.settings.fetch_timeout_seconds) as c:
                res = await c.get(url, params=params, headers=headers)
                res.raise_for_status()
                data = res.json()
        except Exception as exc:
            raise SearchError(f"{type(self).__name__} request failed: {exc}") from exc
        return self._parse(data)[:limit]


class BraveSearch(HttpJsonSearch):  # pragma: no cover - network dependent
    """Brave Search API. Has a free tier, which Tavily's paid depth setting eats.

    Chosen as the free default because it is a documented JSON API with a key,
    rather than a scrape that breaks when someone changes a CSS class.
    """

    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if settings.brave_api_key is None:
            raise SearchError("BRAVE_API_KEY is required for search_provider=brave")
        self._key = settings.brave_api_key.get_secret_value()

    def _request(self, query: str, limit: int) -> tuple[str, dict, dict]:
        return (
            self.ENDPOINT,
            {"q": query, "count": limit},
            {"X-Subscription-Token": self._key, "Accept": "application/json"},
        )

    def _parse(self, data: dict) -> list[SearchHit]:
        return [
            SearchHit(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=(r.get("description") or "")[:500],
            )
            for r in data.get("web", {}).get("results", [])
            if r.get("url")
        ]


class SearxngSearch(HttpJsonSearch):  # pragma: no cover - needs an instance
    """A SearXNG instance you run. No key, no account, no per-query cost.

    The search-side counterpart to the Ollama LLM provider: everything local.
    Most instances ship with JSON disabled, so `search.formats` in settings.yml
    must include `json` or every query returns 403.
    """

    def _request(self, query: str, limit: int) -> tuple[str, dict, dict]:
        base = self.settings.searxng_base_url.rstrip("/")
        return (
            f"{base}/search",
            {"q": query, "format": "json", "categories": "general"},
            {"Accept": "application/json"},
        )

    def _parse(self, data: dict) -> list[SearchHit]:
        return [
            SearchHit(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=(r.get("content") or "")[:500],
            )
            for r in data.get("results", [])
            if r.get("url")
        ]


def build_search(settings: Settings) -> SearchProvider:
    match settings.search_provider:
        case "tavily":
            return TavilySearch(settings)
        case "brave":
            return BraveSearch(settings)
        case "searxng":
            return SearxngSearch(settings)
        case "fake":
            return FakeSearch(settings)
        case other:  # pragma: no cover
            raise SearchError(f"Unknown search_provider: {other}")


# --------------------------------------------------------------------------
# Content fetching
# --------------------------------------------------------------------------

_TAG_STRIP = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\n\s*\n\s*\n+")


def html_to_text(html: str) -> str:
    """Minimal HTML-to-text.

    Deliberately dependency-free. A heavier extractor (trafilatura) is better
    quality but this keeps the install small and the behaviour inspectable,
    which matters more for a portfolio project than a few points of recall.
    """
    text = _TAG_STRIP.sub(" ", html)
    text = _TAGS.sub(" ", text)
    for entity, char in (
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
        ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"),
    ):
        text = text.replace(entity, char)
    text = re.sub(r"[ \t]+", " ", text)
    return _WS.sub("\n\n", text).strip()


class FetchResult(BaseModel):
    source: Source
    text: str


class ContentFetcher:
    """Fetches URLs safely and returns sanitized text.

    Every fetch passes through, in order:
      1. SSRF policy check on EVERY redirect hop (security/ssrf.py)
      2. size + content-type cap
      3. HTML -> text
      4. injection detection + neutralisation (security/sanitize.py)
    """

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self.fetch_count = 0


    async def _get_validating_redirects(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[httpx.Response, str]:
        """GET `url`, re-running the SSRF policy on every redirect hop.

        httpx's own `follow_redirects=True` is unusable for this. It resolves
        and connects to each hop internally, so a 302 to 169.254.169.254 is
        already fetched by the time our code sees a response. Checking only the
        URL we were given is therefore not a guard at all - it guards the one
        URL in the chain an attacker does not need to control.

        `follow_redirects=False` is passed per-request rather than relying on
        the client default, so an injected client (tests, callers) cannot
        silently re-enable httpx's own redirect handling and reopen the hole.

        Returns the final response and the URL that actually served it.
        """
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            assert_url_is_safe(current)
            res = await client.get(current, follow_redirects=False)
            if not res.is_redirect:
                return res, current
            location = res.headers.get("location")
            if not location:
                raise UnsafeURLError(f"Redirect from {current!r} had no Location header")
            # Relative redirects are legal and common. Resolve against the
            # current URL before validating, or a bare "/admin" would be
            # unparseable rather than correctly rejected.
            current = str(httpx.URL(current).join(location))
        raise UnsafeURLError(
            f"Exceeded {MAX_REDIRECTS} redirects starting from {url!r}"
        )

    async def fetch(self, url: str, *, entity_domain: str | None = None) -> FetchResult | None:
        """Return sanitized content, or None if the URL is unsafe/unfetchable.

        Returning None rather than raising is intentional: one bad URL in a
        research run is normal and must not abort the run.
        """
        # Fast path: reject obviously unsafe URLs before building a client.
        # `_get_validating_redirects` re-checks this same URL, which is
        # deliberate - the helper must be safe for any caller, not only this one.
        try:
            assert_url_is_safe(url)
        except UnsafeURLError as exc:
            logger.warning("blocked_unsafe_url", extra={"url": url, "reason": str(exc)})
            return None

        client = self._client or httpx.AsyncClient(
            timeout=self.settings.fetch_timeout_seconds,
            # We drive the redirect chain ourselves. See the helper below.
            follow_redirects=False,
            headers={"User-Agent": self.settings.user_agent},
        )
        owns_client = self._client is None
        try:
            res, final_url = await self._get_validating_redirects(client, url)
            res.raise_for_status()

            ctype = res.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype and ctype:
                logger.info("skipped_non_text", extra={"url": url, "ctype": ctype})
                return None

            raw = res.text[:MAX_RESPONSE_BYTES]
        except UnsafeURLError as exc:
            # A redirect walked somewhere it should not. Same log event as a bad
            # initial URL, because operationally it is the same thing.
            logger.warning("blocked_unsafe_url", extra={"url": url, "reason": str(exc)})
            return None
        except Exception as exc:
            logger.warning("fetch_failed", extra={"url": url, "error": str(exc)})
            return None
        finally:
            if owns_client:
                await client.aclose()

        if final_url != url:
            logger.info("followed_redirect", extra={"from": url, "to": final_url})

        # Everything below describes the page we actually received, so it is
        # keyed on `final_url`. Recording the requested URL instead would let a
        # redirect file its content under a more reputable domain than served it
        # - both a tiering error and a hole in the audit trail.
        url = final_url

        text = html_to_text(raw)
        flags = detect_injection(text)
        clean = neutralise(text)

        if flags:
            logger.warning("injection_detected", extra={"url": url, "flags": flags})

        self.fetch_count += 1
        tier = classify_source(url, entity_domain=entity_domain)
        # A page that tries to manipulate us is not a page we trust.
        if flags and tier is not SourceTier.PRIMARY:
            tier = SourceTier.UNKNOWN

        source = Source(
            # HttpUrl(...) validates here rather than relying on implicit
            # coercion, so a malformed URL fails at the fetch boundary where we
            # can log it, not deep inside the graph.
            url=HttpUrl(url),
            title=(res.headers.get("title") or "")[:300],
            tier=tier,
            content_sha256=Source.hash_content(clean),
            char_count=len(clean),
            injection_flags=flags,
        )
        return FetchResult(source=source, text=clean)

    async def fetch_many(
        self, urls: list[str], *, entity_domain: str | None = None
    ) -> list[FetchResult]:
        """Bounded-concurrency fetch. Failures are dropped, not raised."""
        sem = asyncio.Semaphore(self.settings.fetch_max_concurrency)

        async def one(u: str) -> FetchResult | None:
            async with sem:
                return await self.fetch(u, entity_domain=entity_domain)

        results = await asyncio.gather(*(one(u) for u in urls), return_exceptions=True)
        return [r for r in results if isinstance(r, FetchResult)]
