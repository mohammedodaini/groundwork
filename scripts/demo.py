"""Offline demo: a realistic research run with no API keys and no network.

    python scripts/demo.py

Why this exists: a portfolio repo is usually evaluated by someone who will not
configure API keys. Without this, they see a README and a test suite but never
the actual output. This scripts a plausible run through the real graph - the
same nodes, the same verification, the same critic - with a scripted model.

The scenario itself lives in `groundwork.demo`, so the API replays it too
whenever both providers are `fake`.

IMPORTANT: the LLM responses here are hand-written, not generated. The *pipeline*
is real; the model's answers are fixtures. Every behaviour you see (verbatim
verification, FACT downgrade, injection down-tiering, the research loop) is
produced by the real code paths, not printed from a template.

The demo deliberately includes three things going wrong, because a demo where
everything succeeds teaches nothing:
  1. a fabricated quote that gets caught and downgraded
  2. a page carrying a prompt-injection payload
  3. a criterion the evidence cannot settle -> INSUFFICIENT_EVIDENCE
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundwork.agent.graph import ResearchEngine
from groundwork.config import Settings
from groundwork.demo import DEMO_CRITERIA, DEMO_OBJECTIVE, build_demo_providers
from groundwork.domain.schemas import ResearchRequest

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, BLUE = "\033[32m", "\033[33m", "\033[31m", "\033[34m"

COLOUR = {"FACT": GREEN, "INFERENCE": YELLOW, "UNKNOWN": DIM}


def render(result) -> None:
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}OBJECTIVE{RESET}  {result.request.objective}")
    print(f"{BOLD}{'=' * 78}{RESET}\n")

    print(f"{BOLD}PLAN{RESET}")
    for i, step in enumerate(result.plan, 1):
        print(f"  {i}. {step}")

    print(f"\n{BOLD}SOURCES{RESET}")
    for s in result.sources:
        flag = (
            f"  {RED}[injection: {', '.join(s.injection_flags)}]{RESET}"
            if s.injection_flags
            else ""
        )
        print(f"  {s.tier.value:<10} {s.domain}{flag}")

    ev_by_id = {str(e.id): e for e in result.evidence}
    src_by_id = {str(s.id): s for s in result.sources}

    for ent in result.entities:
        print(f"\n{BOLD}ENTITY  {ent.name}{RESET}")
        if ent.qualification:
            q = ent.qualification
            colour = GREEN if q.decision.value == "QUALIFIED" else YELLOW
            print(f"  {colour}{q.decision.value}{RESET} — {q.rationale}")
            if q.criteria_unknown:
                print(f"  {DIM}unresolved: {', '.join(q.criteria_unknown)}{RESET}")

        print()
        for c in ent.claims:
            colour = COLOUR.get(c.status.value, "")
            verdict = ""
            if c.verdict:
                vc = GREEN if c.verdict.value == "SUPPORTED" else RED
                verdict = f" {vc}[{c.verdict.value}]{RESET}"
            print(f"  {colour}{c.status.value:<9}{RESET}{verdict} {c.text}")
            print(f"            {DIM}confidence {c.confidence:.2f}{RESET}")
            if c.critic_note:
                print(f"            {DIM}{c.critic_note}{RESET}")
            for eid in c.evidence_ids:
                ev = ev_by_id.get(str(eid))
                if not ev:
                    continue
                src = src_by_id.get(str(ev.source_id))
                quote = " ".join(ev.quote.split())[:96]
                mark = f"{GREEN}✓ verified{RESET}" if ev.verbatim_verified else f"{RED}✗{RESET}"
                print(f'            {BLUE}"{quote}"{RESET}')
                print(f"            {DIM}{src.domain if src else '?'} · {mark}{RESET}")
            if not c.evidence_ids:
                print(f"            {DIM}(no evidence attached){RESET}")
            print()

    if result.insights:
        print(f"{BOLD}INSIGHTS{RESET} {DIM}(all INFERENCE by construction){RESET}")
        for i in result.insights:
            print(f"  • {i.text}  {DIM}[{i.confidence:.2f}]{RESET}")

    if result.warnings:
        print(f"\n{BOLD}{YELLOW}WARNINGS{RESET}")
        for w in result.warnings:
            print(f"  ! {w}")

    m = result.metrics
    print(f"\n{BOLD}METRICS{RESET}")
    print(
        f"  entities={len(result.entities)}  claims={len(result.all_claims)}  "
        f"sources={len(result.sources)}  llm_calls={m.llm_calls}  "
        f"latency={m.latency_ms}ms"
    )
    print(
        f"  evidence_coverage={result.evidence_coverage():.0%}  "
        f"unsupported_rate={result.unsupported_claim_rate():.0%}"
    )
    print(f"\n{DIM}Run status would be AWAITING_APPROVAL. Nothing is sent or acted on.{RESET}\n")


async def main() -> None:
    settings = Settings(llm_provider="fake", search_provider="fake", log_level="ERROR")
    llm, search, fetcher = build_demo_providers(settings)
    engine = ResearchEngine(settings=settings, llm=llm, search=search, fetcher=fetcher)
    result = await engine.run(
        ResearchRequest(
            objective=DEMO_OBJECTIVE,
            criteria=list(DEMO_CRITERIA),
            max_entities=3,
            max_search_rounds=2,
        )
    )
    render(result)


if __name__ == "__main__":
    asyncio.run(main())
