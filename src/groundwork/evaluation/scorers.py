"""Evaluation scorers.

DESIGN PRINCIPLE: prefer scorers that are *deterministic* and *checkable* over
scorers that need an LLM judge. An LLM-judged score is itself a model output
with its own error rate; reporting it as "accuracy" without saying so is how
benchmarks become fiction.

We therefore split metrics into two groups and label them in the output:

  INTRINSIC  - computed from the result structure alone. No judge, no ground
               truth. Fully reproducible. (grounding, coverage, cost, latency)
  REFERENCED - compared against a human-written expectation in the dataset.
               Reproducible, but limited by the dataset's coverage.

There is deliberately NO "factual accuracy" metric that asks an LLM whether the
output was true. It would be the most impressive-sounding number here and the
least trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from groundwork.domain.enums import EpistemicStatus
from groundwork.domain.schemas import ResearchResult


@dataclass
class TaskScore:
    task_id: str
    ok: bool = True
    error: str = ""

    # -- intrinsic -------------------------------------------------------
    entities_found: int = 0
    claims_total: int = 0
    facts: int = 0
    inferences: int = 0
    unknowns: int = 0
    evidence_coverage: float = 0.0
    unsupported_rate: float = 0.0
    quote_verification_rate: float = 0.0
    injection_sources_flagged: int = 0

    # -- referenced ------------------------------------------------------
    expected_entities_found: int = 0
    expected_entities_total: int = 0
    forbidden_claims_present: int = 0

    # -- operational -----------------------------------------------------
    latency_ms: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def entity_recall(self) -> float:
        if not self.expected_entities_total:
            return float("nan")
        return self.expected_entities_found / self.expected_entities_total


def _normalise_name(name: str) -> str:
    import re

    name = name.lower()
    # Strip legal suffixes so "Acme B.V." matches "Acme".
    name = re.sub(r"\b(b\.?v\.?|n\.?v\.?|gmbh|ltd|inc|llc|s\.?a\.?)\b", "", name)
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def score_result(
    task_id: str,
    result: ResearchResult,
    *,
    expected_entities: list[str] | None = None,
    forbidden_substrings: list[str] | None = None,
) -> TaskScore:
    """Score one completed research run."""
    claims = result.all_claims
    score = TaskScore(
        task_id=task_id,
        entities_found=len(result.entities),
        claims_total=len(claims),
        facts=sum(1 for c in claims if c.status is EpistemicStatus.FACT),
        inferences=sum(1 for c in claims if c.status is EpistemicStatus.INFERENCE),
        unknowns=sum(1 for c in claims if c.status is EpistemicStatus.UNKNOWN),
        evidence_coverage=result.evidence_coverage(),
        unsupported_rate=result.unsupported_claim_rate(),
        injection_sources_flagged=sum(1 for s in result.sources if s.injection_flags),
        latency_ms=result.metrics.latency_ms,
        llm_calls=result.metrics.llm_calls,
        prompt_tokens=result.metrics.prompt_tokens,
        completion_tokens=result.metrics.completion_tokens,
        estimated_cost_usd=result.metrics.estimated_cost_usd,
        warnings=list(result.warnings),
    )

    if result.evidence:
        verified = sum(1 for e in result.evidence if e.verbatim_verified)
        score.quote_verification_rate = verified / len(result.evidence)

    if expected_entities:
        score.expected_entities_total = len(expected_entities)
        found = {_normalise_name(e.name) for e in result.entities}
        score.expected_entities_found = sum(
            1
            for exp in expected_entities
            if any(_normalise_name(exp) in f or f in _normalise_name(exp) for f in found if f)
        )

    if forbidden_substrings:
        # A "forbidden claim" is a known-false statement the dataset author has
        # seen models produce. Catching these is a hallucination canary.
        blob = " ".join(c.text.lower() for c in claims)
        score.forbidden_claims_present = sum(1 for f in forbidden_substrings if f.lower() in blob)

    return score


@dataclass
class BenchmarkSummary:
    n_tasks: int
    n_ok: int
    mean_evidence_coverage: float
    mean_unsupported_rate: float
    mean_quote_verification_rate: float
    mean_entity_recall: float
    total_forbidden_claims: int
    mean_latency_ms: float
    total_llm_calls: int
    total_cost_usd: float

    def as_markdown(self) -> str:
        rows = [
            ("Tasks run", f"{self.n_tasks}"),
            ("Tasks completed without error", f"{self.n_ok}/{self.n_tasks}"),
            (
                "Mean evidence coverage (FACTs with verified quote)",
                f"{self.mean_evidence_coverage:.2%}",
            ),
            ("Mean unsupported-claim rate (critic)", f"{self.mean_unsupported_rate:.2%}"),
            ("Mean quote verification rate", f"{self.mean_quote_verification_rate:.2%}"),
            ("Mean expected-entity recall", f"{self.mean_entity_recall:.2%}"),
            ("Known-false claims produced", f"{self.total_forbidden_claims}"),
            ("Mean latency per task", f"{self.mean_latency_ms:.0f} ms"),
            ("Total LLM calls", f"{self.total_llm_calls}"),
            ("Estimated total cost", f"${self.total_cost_usd:.4f}"),
        ]
        out = ["| Metric | Value |", "| --- | --- |"]
        out += [f"| {k} | {v} |" for k, v in rows]
        return "\n".join(out)


def summarise(scores: list[TaskScore]) -> BenchmarkSummary:
    import math

    ok = [s for s in scores if s.ok]
    n = max(len(ok), 1)
    recalls = [s.entity_recall for s in ok if not math.isnan(s.entity_recall)]
    return BenchmarkSummary(
        n_tasks=len(scores),
        n_ok=len(ok),
        mean_evidence_coverage=sum(s.evidence_coverage for s in ok) / n,
        mean_unsupported_rate=sum(s.unsupported_rate for s in ok) / n,
        mean_quote_verification_rate=sum(s.quote_verification_rate for s in ok) / n,
        mean_entity_recall=(sum(recalls) / len(recalls)) if recalls else 0.0,
        total_forbidden_claims=sum(s.forbidden_claims_present for s in ok),
        mean_latency_ms=sum(s.latency_ms for s in ok) / n,
        total_llm_calls=sum(s.llm_calls for s in ok),
        total_cost_usd=sum(s.estimated_cost_usd for s in ok),
    )
