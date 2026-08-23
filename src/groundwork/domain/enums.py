"""Enumerations shared across the domain model.

The single most important idea in this codebase lives here: `EpistemicStatus`.

Most "AI research agents" return prose. Prose cannot be audited, cannot be
scored, and cannot be trusted. We instead force the system to emit *claims*,
each of which must declare how it is known:

    FACT      - directly supported by a verbatim span in a retrieved source
    INFERENCE - reasoned from evidence, but not stated by any source
    UNKNOWN   - the system looked and could not establish this

This distinction is what makes Milestone 9 (evaluation) possible at all: you
cannot measure "hallucination rate" unless the system commits, per claim, to
whether it believes something is sourced or reasoned.
"""

from __future__ import annotations

from enum import StrEnum


class EpistemicStatus(StrEnum):
    """How a claim is known. See module docstring."""

    FACT = "FACT"
    INFERENCE = "INFERENCE"
    UNKNOWN = "UNKNOWN"


class SourceTier(StrEnum):
    """Coarse source-quality tiers.

    Deliberately coarse. A finer-grained score would imply a precision we
    cannot justify, and would be harder to defend in an interview than a
    rule you can state in one sentence.
    """

    PRIMARY = "PRIMARY"  # the entity's own site, official registries, filings
    REPUTABLE = "REPUTABLE"  # established news, trade press, academic
    SECONDARY = "SECONDARY"  # aggregators, directories, blogs
    UNKNOWN = "UNKNOWN"  # unclassifiable


class JobStatus(StrEnum):
    """Lifecycle of a research job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class VerificationVerdict(StrEnum):
    """Outcome of the critic stage for a single claim."""

    SUPPORTED = "SUPPORTED"  # evidence backs the claim
    UNSUPPORTED = "UNSUPPORTED"  # no evidence backs the claim
    CONTRADICTED = "CONTRADICTED"  # evidence actively conflicts
    OVERSTATED = "OVERSTATED"  # evidence is weaker than the claim implies


class QualificationDecision(StrEnum):
    """Whether an entity meets the user's stated criteria."""

    QUALIFIED = "QUALIFIED"
    NOT_QUALIFIED = "NOT_QUALIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
