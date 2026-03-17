"""
Pydantic models for Feature 4 (Methodology Comparison).

Defines request/response schemas for comparing methodologies across
2-4 papers selected from a citation map.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class MethodologyCompareRequest(BaseModel):
    """Request body for methodology comparison."""

    work_ids: list[str]

    @field_validator("work_ids")
    @classmethod
    def validate_work_ids(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("At least 2 work_ids are required for comparison")
        if len(v) > 4:
            raise ValueError("At most 4 work_ids can be compared at once")
        if len(v) != len(set(v)):
            raise ValueError("work_ids must be unique")
        return v


# ---------------------------------------------------------------------------
# Methodology Fingerprint (per paper, cached)
# ---------------------------------------------------------------------------

class MethodologyFingerprint(BaseModel):
    """Structured methodology profile extracted from a paper."""

    approach: Optional[str] = None
    data_requirements: Optional[str] = None
    assumptions: list[str] = []
    validation_method: Optional[str] = None
    limitations: list[str] = []
    domain: Optional[str] = None
    key_components: list[str] = []
    novelty_over_prior: Optional[str] = None
    validation_metrics: list[dict] = []


class PaperMethodProfile(BaseModel):
    """A paper with its extracted methodology fingerprint."""

    work_id: str
    title: str
    year: Optional[int] = None
    source_quality: Literal["full_text", "abstract_only"]
    methodology_fingerprint: MethodologyFingerprint


# ---------------------------------------------------------------------------
# Citation Lineage
# ---------------------------------------------------------------------------

class DirectCitation(BaseModel):
    """A direct citation between two selected papers."""

    from_work_id: str
    to_work_id: str


class SharedReference(BaseModel):
    """A paper referenced by multiple selected papers."""

    work_id: str
    title: str
    cited_by: list[str]  # which of the selected papers cite this


class CitationLineage(BaseModel):
    """Citation relationships between the selected papers."""

    direct_citations: list[DirectCitation] = []
    shared_references: list[SharedReference] = []
    evolution_chain: Optional[str] = None  # e.g. "W123 (2012) → W456 (2017) → W789 (2022)"


# ---------------------------------------------------------------------------
# Convergence/Divergence Analysis (paradigm-centered)
# ---------------------------------------------------------------------------

class CommonProblem(BaseModel):
    """The shared problem that all compared papers address."""

    domain: str  # e.g. "computer vision", "NLP"
    challenge: str  # The specific problem (e.g. "image classification at scale")
    why_hard: str  # What makes this problem difficult


class Paradigm(BaseModel):
    """A methodological paradigm/approach cluster."""

    name: str  # e.g. "Statistical methods", "Deep learning", "Simulation-based"
    papers: list[str]  # work_ids that use this paradigm
    mechanism: str  # How this paradigm tackles the problem
    philosophy: str  # Underlying assumptions/worldview


class ConvergenceDivergence(BaseModel):
    """Analysis of how papers converge on the problem but diverge in approach."""

    common_problem: CommonProblem
    paradigms: list[Paradigm] = []
    divergence_summary: str  # High-level summary of the competing approaches


# ---------------------------------------------------------------------------
# Strengths/Weaknesses Matrix (paper-centered)
# ---------------------------------------------------------------------------

class Capability(BaseModel):
    """Something a methodology handles well."""

    capability: str  # What it can do
    mechanism: str  # How it achieves this
    evidence: str  # Proof/benchmark/result


class Limitation(BaseModel):
    """Something a methodology struggles with."""

    limitation: str  # What it can't handle
    cause: str  # Why (design choice or assumption)
    consequence: str  # Impact on use cases


class Assumption(BaseModel):
    """A key assumption the methodology makes."""

    assumption: str  # The assumption
    if_violated: str  # What happens if this doesn't hold


class Complement(BaseModel):
    """How another paper covers this paper's weakness."""

    other_work_id: str
    other_title: Optional[str] = None  # Title (for external papers not in comparison set)
    other_year: Optional[int] = None  # Year (for external papers not in comparison set)
    coverage: str  # How the other paper addresses the gap


class PaperStrengthsWeaknesses(BaseModel):
    """Structured strengths/weaknesses for one paper."""

    work_id: str
    title: str
    handles_well: list[Capability] = []
    struggles_with: list[Limitation] = []
    assumptions: list[Assumption] = []
    complemented_by: list[Complement] = []


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class DecisionScenario(BaseModel):
    """A specific scenario and which paper to use."""

    scenario: str  # The use case or context
    use: str  # work_id to use
    why: str  # Reasoning


class Recommendation(BaseModel):
    """LLM's recommendation for researchers."""

    summary: str  # Quick take on the landscape
    decision_matrix: list[DecisionScenario] = []
    can_combine: bool = False  # Can these methods be used together?
    combination_notes: Optional[str] = None  # How to combine if applicable


# ---------------------------------------------------------------------------
# Top-level Response
# ---------------------------------------------------------------------------

class MethodologyComparisonResponse(BaseModel):
    """Complete response for methodology comparison."""

    work_ids: list[str]
    papers: list[PaperMethodProfile]
    paper_index: dict[str, int] = {}  # work_id → 1-based number, e.g. {"W123": 1, "W456": 2}
    referenced_works: dict[str, str] = {}  # work_id -> title for external papers mentioned
    lineage: CitationLineage
    convergence_divergence: ConvergenceDivergence
    strengths_weaknesses_matrix: list[PaperStrengthsWeaknesses]
    recommendation: Recommendation
    confidence: Literal["low", "medium", "high"]
