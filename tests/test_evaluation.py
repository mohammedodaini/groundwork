"""Tests for the evaluation harness.

The evaluator is the part of the project a reviewer trusts least by default,
because it is the part that produces the numbers. So it is tested like
production code: the scorers must be correct, and the dataset must be valid.
"""

from __future__ import annotations

import pytest

from groundwork.domain.enums import EpistemicStatus, VerificationVerdict
from groundwork.domain.schemas import (
    Claim,
    Entity,
    Evidence,
    ResearchRequest,
    ResearchResult,
    RunMetrics,
    Source,
)
from groundwork.evaluation.run_eval import DEFAULT_DATASET, load_tasks, render_report
from groundwork.evaluation.scorers import _normalise_name, score_result, summarise


def build_result(
    *, facts: int = 1, verified: bool = True, claim_text: str = "Acme employs 45 staff."
) -> ResearchResult:
    source = Source(url="https://example.com/a")
    evidence, claims = [], []
    for _ in range(facts):
        ev = Evidence(source_id=source.id, quote="a quote long enough", verbatim_verified=verified)
        evidence.append(ev)
        claims.append(
            Claim(
                text=claim_text,
                status=EpistemicStatus.FACT,
                evidence_ids=[ev.id],
                verdict=VerificationVerdict.SUPPORTED,
            )
        )
    return ResearchResult(
        request=ResearchRequest(objective="A sufficiently long research objective."),
        entities=[Entity(name="Acme B.V.", claims=claims)],
        sources=[source],
        evidence=evidence,
        metrics=RunMetrics(latency_ms=100, llm_calls=3, estimated_cost_usd=0.01),
    )


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------


def test_scores_fully_grounded_result() -> None:
    score = score_result("t1", build_result())
    assert score.evidence_coverage == 1.0
    assert score.quote_verification_rate == 1.0
    assert score.unsupported_rate == 0.0
    assert score.facts == 1


def test_unverified_quotes_lower_the_score() -> None:
    """A result full of unverifiable quotes must score badly, not well."""
    score = score_result("t1", build_result(verified=False))
    assert score.quote_verification_rate == 0.0
    assert score.evidence_coverage == 0.0


def test_entity_recall_matches_across_legal_suffixes() -> None:
    """'Acme B.V.' should match an expectation written as 'Acme'."""
    score = score_result("t1", build_result(), expected_entities=["Acme"])
    assert score.expected_entities_found == 1
    assert score.entity_recall == 1.0


def test_missing_expected_entity_is_recorded() -> None:
    score = score_result("t1", build_result(), expected_entities=["Acme", "Globex"])
    assert score.expected_entities_found == 1
    assert score.entity_recall == 0.5


def test_forbidden_substring_is_detected() -> None:
    """The hallucination canary."""
    result = build_result(claim_text="Acme was founded in 1776 in Boston.")
    score = score_result("t1", result, forbidden_substrings=["founded in 1776"])
    assert score.forbidden_claims_present == 1


def test_forbidden_substring_absent_scores_zero() -> None:
    score = score_result("t1", build_result(), forbidden_substrings=["founded in 1776"])
    assert score.forbidden_claims_present == 0


def test_entity_recall_is_nan_without_expectations() -> None:
    """No ground truth must not silently become 0% recall."""
    import math

    score = score_result("t1", build_result())
    assert math.isnan(score.entity_recall)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Acme B.V.", "acme"),
        ("Acme  Holdings   N.V.", "acme holdings"),
        ("ASML Holding NV", "asml holding"),
    ],
)
def test_normalise_name(raw: str, expected: str) -> None:
    assert _normalise_name(raw) == expected


def test_summary_excludes_failed_runs_from_means() -> None:
    from groundwork.evaluation.scorers import TaskScore

    scores = [
        score_result("ok", build_result()),
        TaskScore(task_id="bad", ok=False, error="boom"),
    ]
    summary = summarise(scores)
    assert summary.n_tasks == 2
    assert summary.n_ok == 1
    assert summary.mean_evidence_coverage == 1.0  # not diluted by the failure


def test_summary_renders_markdown_table() -> None:
    md = summarise([score_result("t1", build_result())]).as_markdown()
    assert md.startswith("| Metric | Value |")
    assert "evidence coverage" in md.lower()


# --------------------------------------------------------------------------
# Dataset integrity
# --------------------------------------------------------------------------


def test_benchmark_dataset_is_valid() -> None:
    tasks = load_tasks(DEFAULT_DATASET)
    assert len(tasks) >= 8
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    for t in tasks:
        assert len(t["objective"]) >= 10
        assert isinstance(t.get("criteria", []), list)
        assert 1 <= t.get("max_search_rounds", 2) <= 6
        # Every task must document why it exists.
        assert t.get("notes"), f"task {t['id']} has no notes"


def test_dataset_includes_adversarial_tasks() -> None:
    """The set must contain tasks the system is expected to FAIL to answer."""
    ids = {t["id"] for t in load_tasks(DEFAULT_DATASET)}
    assert "nonexistent-company" in ids
    assert "contradictory-figures" in ids


def test_report_labels_fake_provider_results(settings) -> None:
    """A report generated with the fake provider must say so, loudly."""
    tasks = load_tasks(DEFAULT_DATASET)[:1]
    report = render_report([score_result("t1", build_result())], settings, tasks, 1)
    assert "fake" in report.lower()
    # The disclaimer must survive rewording, so assert its substance: the run
    # is labelled fake, it disclaims research quality, and it tells the reader
    # to re-run against a real provider.
    lowered = report.lower()
    assert "nothing about" in lowered and "research quality" in lowered
    assert "real provider" in lowered


def test_report_contains_no_invented_numbers(settings) -> None:
    """Everything in the table must trace back to a supplied score."""
    score = score_result("t1", build_result())
    report = render_report([score], settings, load_tasks(DEFAULT_DATASET)[:1], 1)
    assert "Tasks run | 1" in report
    assert "`t1`" in report
