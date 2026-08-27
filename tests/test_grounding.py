"""Tests for the grounding guarantees.

These verify the project's central claim: that a fabricated quote cannot become
a FACT, and that a claim can never cite evidence that does not exist.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from groundwork.agent.nodes.extract import _normalise, verify_quote
from groundwork.domain.enums import EpistemicStatus, SourceTier, VerificationVerdict
from groundwork.domain.schemas import (
    Claim,
    Entity,
    Evidence,
    ResearchRequest,
    ResearchResult,
    Source,
)

PAGE = (
    "Acme Technical Wholesale B.V. is based in Arnhem, Gelderland.\n\n"
    "The company   was founded in 1998 and employs 45 staff."
)


# --------------------------------------------------------------------------
# Verbatim verification
# --------------------------------------------------------------------------


def test_exact_quote_verifies() -> None:
    ok, offset = verify_quote("founded in 1998 and employs 45 staff", PAGE)
    assert ok
    assert offset is not None


def test_quote_with_different_whitespace_verifies() -> None:
    """HTML-to-text mangles whitespace; that is our artefact, not the model's."""
    ok, _ = verify_quote("The company was founded in 1998", PAGE)
    assert ok


def test_quote_with_curly_quotes_verifies() -> None:
    page = "The CEO said \u201cwe are expanding into Germany\u201d last year."
    ok, _ = verify_quote('"we are expanding into Germany"', page)
    assert ok


def test_fabricated_quote_is_rejected() -> None:
    """The core anti-hallucination test."""
    ok, offset = verify_quote("The company employs 500 staff worldwide", PAGE)
    assert not ok
    assert offset is None


def test_paraphrased_quote_is_rejected() -> None:
    """Paraphrase is not evidence, even when it is semantically true."""
    ok, _ = verify_quote("Acme has forty-five employees", PAGE)
    assert not ok


def test_truncated_quote_that_changes_meaning_is_rejected() -> None:
    ok, _ = verify_quote("employs 45000 staff", PAGE)
    assert not ok


def test_trivially_short_quote_is_rejected() -> None:
    """Short strings match by accident; they are not evidence."""
    ok, _ = verify_quote("the", PAGE)
    assert not ok


def test_normalise_is_idempotent() -> None:
    once = _normalise(PAGE)
    assert _normalise(once) == once


# --------------------------------------------------------------------------
# Claim invariants
# --------------------------------------------------------------------------


def test_fact_without_evidence_is_auto_downgraded() -> None:
    claim = Claim(text="Acme has 45 staff.", status=EpistemicStatus.FACT, evidence_ids=[])
    assert claim.status is EpistemicStatus.INFERENCE
    assert "auto-downgraded" in claim.critic_note


def test_fact_with_evidence_survives() -> None:
    claim = Claim(
        text="Acme has 45 staff.",
        status=EpistemicStatus.FACT,
        evidence_ids=[uuid4()],
    )
    assert claim.status is EpistemicStatus.FACT


def test_inference_without_evidence_is_allowed() -> None:
    """INFERENCE is explicitly permitted to lack a quote - that is its meaning."""
    claim = Claim(text="Acme likely benefits from automation.", status=EpistemicStatus.INFERENCE)
    assert claim.status is EpistemicStatus.INFERENCE


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim(text="x" * 5, status=EpistemicStatus.INFERENCE, confidence=1.5)


def test_extra_fields_are_forbidden() -> None:
    """LLMs invent keys; we want that loud, not silent."""
    with pytest.raises(ValidationError):
        Claim.model_validate({"text": "hello there", "status": "FACT", "hallucinated_field": 1})


# --------------------------------------------------------------------------
# Referential integrity
# --------------------------------------------------------------------------


def _result_with(evidence_ids: list, evidence: list, sources: list) -> ResearchResult:
    return ResearchResult(
        request=ResearchRequest(objective="A sufficiently long research objective."),
        entities=[
            Entity(
                name="Acme",
                claims=[
                    Claim(
                        text="Acme exists.",
                        status=EpistemicStatus.FACT,
                        evidence_ids=evidence_ids,
                    )
                ],
            )
        ],
        sources=sources,
        evidence=evidence,
    )


def test_dangling_evidence_reference_is_rejected() -> None:
    source = Source(url="https://example.com/a")
    with pytest.raises(ValidationError, match="unknown evidence"):
        _result_with([uuid4()], [], [source])


def test_evidence_pointing_at_unknown_source_is_rejected() -> None:
    ev = Evidence(source_id=uuid4(), quote="a real quote here")
    with pytest.raises(ValidationError, match="unknown source"):
        _result_with([ev.id], [ev], [])


def test_valid_graph_passes() -> None:
    source = Source(url="https://example.com/a")
    ev = Evidence(source_id=source.id, quote="a real quote here", verbatim_verified=True)
    result = _result_with([ev.id], [ev], [source])
    assert result.evidence_coverage() == 1.0


# --------------------------------------------------------------------------
# Derived metrics
# --------------------------------------------------------------------------


def test_unsupported_rate_counts_all_bad_verdicts() -> None:
    source = Source(url="https://example.com/a")
    ev = Evidence(source_id=source.id, quote="a real quote here", verbatim_verified=True)
    result = ResearchResult(
        request=ResearchRequest(objective="A sufficiently long research objective."),
        sources=[source],
        evidence=[ev],
        entities=[
            Entity(
                name="Acme",
                claims=[
                    Claim(
                        text="ok claim",
                        status=EpistemicStatus.FACT,
                        evidence_ids=[ev.id],
                        verdict=VerificationVerdict.SUPPORTED,
                    ),
                    Claim(
                        text="bad claim",
                        status=EpistemicStatus.INFERENCE,
                        verdict=VerificationVerdict.UNSUPPORTED,
                    ),
                    Claim(
                        text="loud claim",
                        status=EpistemicStatus.INFERENCE,
                        verdict=VerificationVerdict.OVERSTATED,
                    ),
                ],
            )
        ],
    )
    assert result.unsupported_claim_rate() == pytest.approx(2 / 3)


def test_source_domain_is_derived_and_normalised() -> None:
    assert Source(url="https://www.Example.com/x").domain == "example.com"


def test_source_tier_defaults_to_unknown() -> None:
    assert Source(url="https://example.com/x").tier is SourceTier.UNKNOWN
