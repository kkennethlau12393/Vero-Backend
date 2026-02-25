"""
Schemas for Feature 5: Research Gap Analysis.

This feature provides comprehensive analysis of research gaps in a
research area, identifying unexplored directions and validating them
against external sources.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from uuid import UUID


# ============================================================================
# Gap Types and Explanations
# ============================================================================

GapType = Literal["structural", "coverage", "temporal", "methodological", "novelty"]

GAP_TYPE_EXPLANATIONS: dict[GapType, str] = {
    "structural": "Research clusters that share related goals but lack citation connections",
    "coverage": "Topics mentioned in the research area but underrepresented in papers",
    "temporal": "Time periods with missing or stagnant research activity",
    "methodological": "Methods that haven't been combined or applied to certain domains",
    "novelty": "Areas dominated by incremental work, lacking pioneering contributions",
}


# ============================================================================
# Coverage Tracking
# ============================================================================

class CoverageBreakdown(BaseModel):
    """Breakdown of coverage percentage by source."""
    map_wide_timeline: float = 0.0  # Up to 15%
    methodology_comparisons: float = 0.0  # Up to 30% (with stale penalty)
    node_specific_exploration: float = 0.0  # Up to 28% (4% per unique action)
    methodology_comparison_count: int = 0
    methodology_stale_penalty: float = 0.0  # Total penalty applied
    node_specific_count: int = 0  # Unique (work_id, activity_type) pairs
    novelty_count: int = 0
    nodes_explored: int = 0
    total_nodes: int = 0
    entry_point: Optional[str] = None  # "rank" | "citation_map" | None


class GapAnalysisStatus(BaseModel):
    """Status of gap analysis availability for a map or rank job."""
    unlocked: bool
    coverage_pct: float
    breakdown: CoverageBreakdown
    message: str  # e.g., "Explored 35% of research area. Explore more to unlock."


# ============================================================================
# Activity Summary
# ============================================================================

class MethodologyActivityDetail(BaseModel):
    """Detail for methodology comparison activities."""
    count: int = 0
    by_node_count: dict[str, int] = {}  # {"2": 1, "3": 2, "4": 0}
    work_ids_compared: list[str] = []


class NodeActivityDetail(BaseModel):
    """Detail for per-node activities (novelty, timeline, details)."""
    count: int = 0
    work_ids: list[str] = []


class SimpleActivityDetail(BaseModel):
    """Detail for simple count-only activities."""
    count: int = 0


class ActivitySummary(BaseModel):
    """Summary of all research activities for a map or rank job."""
    entry_point: Optional[str] = None  # "rank" | "citation_map"
    activities: dict[str, Any] = {}
    coverage_pct: float = 0.0
    first_activity_at: Optional[str] = None
    last_activity_at: Optional[str] = None


# ============================================================================
# Evidence and Gap Cards
# ============================================================================

class Evidence(BaseModel):
    """A paper that provides evidence for a gap."""
    work_id: str
    authors: str  # e.g., "Zoph & Le" or "Howard et al."
    year: int
    title: str
    role: str  # e.g., "Cluster A representative", "Rare bridge paper"


class ExternalValidationResult(BaseModel):
    """Result from GPT-5.2 + web_search validation."""
    status: Literal["open", "partial", "addressed"]
    reasoning: str
    sources: list[dict[str, Any]] = []  # External sources found
    coverage_pct: Optional[float] = None  # 0-100, GPT's estimate of how much of the gap is addressed


class GapCard(BaseModel):
    """A detected research gap with evidence and suggested direction."""
    gap_id: str  # Generated ID (e.g., "gap_1")
    type: GapType
    type_explanation: str  # From GAP_TYPE_EXPLANATIONS
    title: str
    description: str  # Long paragraph with inline citations (Author, Year)
    evidence: list[Evidence] = []  # Full list of papers supporting this gap
    suggested_direction: str
    confidence: float = Field(ge=0.0, le=1.0)  # Computed percentage
    validation_result: Optional[ExternalValidationResult] = None
    detection_score: float = 0.0  # Internal score from heuristic detectors
    data_sources_used: list[str] = []  # Which internal data sources informed this gap


# ============================================================================
# Gap Detection Candidates (Internal)
# ============================================================================

class StructuralGapCandidate(BaseModel):
    """Candidate structural gap from citation graph analysis."""
    cluster_a_id: str
    cluster_b_id: str
    cluster_a_label: str
    cluster_b_label: str
    topic_similarity: float
    citation_density: float
    gap_score: float
    representative_papers_a: list[str] = []  # work_ids
    representative_papers_b: list[str] = []  # work_ids


class CoverageGapCandidate(BaseModel):
    """Candidate coverage gap from ranked list analysis."""
    topic: str
    expected_in_query: bool
    coverage_ratio: float
    gap_score: float
    related_papers: list[str] = []  # work_ids


class TemporalGapCandidate(BaseModel):
    """Candidate temporal gap from timeline analysis."""
    start_year: int
    end_year: int
    period_label: str  # e.g., "2018-2021"
    publication_rate: float
    average_rate: float
    gap_score: float
    is_comeback: bool = False  # active -> stagnant -> active pattern


class MethodologicalGapCandidate(BaseModel):
    """Candidate methodological gap from methodology comparisons."""
    method: str
    domain: str
    neighbor_density: float  # How populated adjacent cells are
    gap_score: float
    related_method_papers: list[str] = []  # work_ids using this method
    related_domain_papers: list[str] = []  # work_ids in this domain


class NoveltyGapCandidate(BaseModel):
    """Candidate novelty gap from novelty assessments."""
    cluster_id: str
    cluster_label: str
    avg_novelty: float
    max_novelty: float
    gap_score: float
    papers_in_cluster: list[str] = []  # work_ids


# ============================================================================
# API Request/Response
# ============================================================================

class GapAnalysisResponse(BaseModel):
    """Response for gap analysis endpoint."""
    job_id: UUID
    gaps: list[GapCard] = []
    coverage_pct: float
    data_sources_used: list[str] = []  # ["citation_graph", "timeline", "methodology", ...]
    total_candidates_detected: int = 0
    candidates_validated: int = 0


class GapAnalysisJobStatus(BaseModel):
    """Status of an async gap analysis job."""
    job_id: UUID
    status: Literal["pending", "detecting", "validating", "complete", "failed"]
    progress: float = 0.0  # 0.0 to 1.0
    message: Optional[str] = None
    result: Optional[GapAnalysisResponse] = None
