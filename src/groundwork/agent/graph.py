"""LangGraph workflow assembly.

    plan -> gather -> extract -> critic -> reflect
                        ^                    |
                        |____ (needs more) __|
                                             |
                                     (sufficient)
                                             v
                                   qualify -> insight -> END

The single conditional edge out of `reflect` is what makes this an agent rather
than a pipeline: the system decides at runtime whether to search again, and it
emits *targeted* queries aimed at the gap it identified.

Why LangGraph rather than hand-rolling this (ADR-001): the loop needs durable,
inspectable state and a checkpointer so a run can pause for human approval and
resume later. Writing that correctly by hand is more code than the graph, and
the graph gives us per-node tracing for free.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC
from functools import partial

from langgraph.graph import END, StateGraph

from groundwork.agent.nodes.critic import critic_node
from groundwork.agent.nodes.extract import extract_node
from groundwork.agent.nodes.gather import gather_node
from groundwork.agent.nodes.plan import plan_node
from groundwork.agent.nodes.reflect import insight_node, qualify_node, reflect_node
from groundwork.agent.state import ResearchState, initial_state
from groundwork.config import Settings
from groundwork.domain.schemas import ResearchRequest, ResearchResult, RunMetrics
from groundwork.observability.tracing import RunTracer
from groundwork.providers.llm import LLMProvider
from groundwork.providers.search import ContentFetcher, SearchProvider

logger = logging.getLogger(__name__)


def route_after_reflect(state: ResearchState) -> str:
    """Conditional edge: loop back to gathering, or move on to synthesis."""
    return "gather" if state.get("should_continue_research") else "qualify"


def build_graph(
    *,
    llm: LLMProvider,
    search: SearchProvider,
    fetcher: ContentFetcher,
    tracer: RunTracer | None = None,
):
    """Compile the workflow.

    Dependencies are injected via `partial` rather than read from globals, so
    tests can pass fakes and the graph has no hidden state.
    """
    tracer = tracer or RunTracer()

    def traced(name: str, fn, *, error_update: dict | None = None):
        """Wrap a node with tracing and fail-closed error containment.

        `error_update` is essential and was a real bug during development: if a
        node that drives a conditional edge raises, returning only
        {"errors": [...]} leaves the routing flag at its previous value. For
        `reflect`, that value is `should_continue_research=True`, so the graph
        looped until it hit the recursion limit - a failing node caused an
        infinite research loop instead of a clean stop.

        The rule this encodes: error containment must also reset any state the
        router depends on. Failing closed beats failing open, especially when
        failing open costs money per iteration.
        """

        async def wrapper(state: ResearchState) -> dict:
            with tracer.span(name):
                try:
                    return await fn(state)
                except Exception as exc:
                    logger.exception("node_failed", extra={"node": name})
                    tracer.record_error(name, str(exc))
                    update = {"errors": [f"{name}: {exc}"]}
                    if error_update:
                        update.update(error_update)
                    return update

        return wrapper

    graph = StateGraph(ResearchState)

    graph.add_node("plan", traced("plan", partial(plan_node, llm=llm)))
    graph.add_node(
        "gather", traced("gather", partial(gather_node, search=search, fetcher=fetcher))
    )
    graph.add_node("extract", traced("extract", partial(extract_node, llm=llm)))
    graph.add_node("critic", traced("critic", partial(critic_node, llm=llm)))
    graph.add_node(
        "reflect",
        traced(
            "reflect",
            partial(reflect_node, llm=llm),
            # Fail closed: a broken reflect node stops research, never loops.
            error_update={
                "should_continue_research": False,
                "gap_reason": "Reflection node failed; stopping research.",
            },
        ),
    )
    graph.add_node("qualify", traced("qualify", partial(qualify_node, llm=llm)))
    graph.add_node("insight", traced("insight", partial(insight_node, llm=llm)))

    graph.set_entry_point("plan")
    graph.add_edge("plan", "gather")
    graph.add_edge("gather", "extract")
    graph.add_edge("extract", "critic")
    graph.add_edge("critic", "reflect")
    graph.add_conditional_edges(
        "reflect", route_after_reflect, {"gather": "gather", "qualify": "qualify"}
    )
    graph.add_edge("qualify", "insight")
    graph.add_edge("insight", END)

    return graph.compile()


class ResearchEngine:
    """Thin façade over the compiled graph.

    Owns metric collection and the conversion from mutable graph state to an
    immutable, validated `ResearchResult`. Keeping this out of the graph means
    the graph stays testable node-by-node.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMProvider,
        search: SearchProvider,
        fetcher: ContentFetcher,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.search = search
        self.fetcher = fetcher
        self.tracer = RunTracer()
        self.graph = build_graph(llm=llm, search=search, fetcher=fetcher, tracer=self.tracer)

    async def run(self, request: ResearchRequest) -> ResearchResult:
        from datetime import datetime

        # Two clocks on purpose: perf_counter for a monotonic duration that is
        # immune to NTP adjustments, wall clock for human-readable timestamps.
        started = time.perf_counter()
        started_at = datetime.now(UTC)
        state = initial_state(request)

        # `recursion_limit` is a second, framework-level guard against runaway
        # loops, independent of our own `max_search_rounds`.
        final: ResearchState = await self.graph.ainvoke(
            state, config={"recursion_limit": 50}
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        metrics = RunMetrics(
            started_at=started_at,
            finished_at=datetime.now(UTC),
            latency_ms=elapsed_ms,
            llm_calls=self.llm.call_count,
            prompt_tokens=self.llm.prompt_tokens,
            completion_tokens=self.llm.completion_tokens,
            search_calls=self.search.call_count,
            pages_fetched=self.fetcher.fetch_count,
            estimated_cost_usd=round(self.llm.estimated_cost_usd(), 6),
            errors=list(final.get("errors", [])),
        )

        result = ResearchResult(
            request=request,
            plan=final.get("plan", []),
            entities=final.get("entities", []),
            sources=final.get("sources", []),
            evidence=final.get("evidence", []),
            insights=final.get("insights", []),
            metrics=metrics,
            warnings=list(final.get("warnings", [])),
        )
        logger.info(
            "run_complete",
            extra={
                "entities": len(result.entities),
                "claims": len(result.all_claims),
                "latency_ms": elapsed_ms,
            },
        )
        return result
