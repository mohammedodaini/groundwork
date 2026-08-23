"""Gathering node: run pending queries, fetch pages, deduplicate.

Entirely deterministic. No LLM is involved in deciding *how* to fetch, only in
deciding *what* to look for (plan node) and what it means (extract node).

Deduplication happens on two axes:
  - URL, to avoid refetching within a run
  - content SHA-256, to collapse the same article served on mirror domains
The second one matters more than it looks: without it, three syndicated copies
of one press release read like three independent confirmations, which inflates
apparent evidence strength.
"""

from __future__ import annotations

import logging

from groundwork.agent.state import ResearchState
from groundwork.providers.search import ContentFetcher, SearchError, SearchProvider

logger = logging.getLogger(__name__)

MAX_URLS_PER_QUERY = 4


async def gather_node(
    state: ResearchState,
    *,
    search: SearchProvider,
    fetcher: ContentFetcher,
) -> dict:
    """Execute every query not yet executed, fetch the results."""
    pending = [q for q in state.get("queries", []) if q not in set(state.get("executed_queries", []))]
    if not pending:
        return {"warnings": ["Gather called with no pending queries."]}

    seen_urls = {str(s.url) for s in state.get("sources", [])}
    seen_hashes = {s.content_sha256 for s in state.get("sources", []) if s.content_sha256}

    candidate_urls: list[str] = []
    errors: list[str] = []

    for query in pending:
        try:
            hits = await search.search(query, limit=MAX_URLS_PER_QUERY)
        except SearchError as exc:
            # One failed query must not kill the run.
            logger.warning("search_failed", extra={"query": query, "error": str(exc)})
            errors.append(f"Search failed for {query!r}: {exc}")
            continue

        for hit in hits:
            if hit.url not in seen_urls and hit.url not in candidate_urls:
                candidate_urls.append(hit.url)

    if not candidate_urls:
        return {
            "executed_queries": pending,
            "errors": errors,
            "warnings": ["No new URLs found for this round."],
        }

    results = await fetcher.fetch_many(candidate_urls)

    new_sources = []
    new_texts: dict[str, str] = dict(state.get("source_texts", {}))
    duplicates = 0

    for res in results:
        if res.source.content_sha256 in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(res.source.content_sha256)
        new_sources.append(res.source)
        new_texts[str(res.source.id)] = res.text

    warnings: list[str] = []
    if duplicates:
        warnings.append(f"Dropped {duplicates} duplicate page(s) by content hash.")

    flagged = [s for s in new_sources if s.injection_flags]
    if flagged:
        warnings.append(
            f"{len(flagged)} page(s) contained prompt-injection patterns and were down-tiered: "
            + ", ".join(s.domain for s in flagged)
        )

    logger.info(
        "gather_complete",
        extra={"queries": len(pending), "fetched": len(new_sources)},
    )

    return {
        "executed_queries": pending,
        "sources": new_sources,
        "source_texts": new_texts,
        "errors": errors,
        "warnings": warnings,
    }
