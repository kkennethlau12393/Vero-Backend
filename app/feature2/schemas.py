from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from uuid import UUID

Grouping = Literal["topic", "subfield"]

class BuildMapRequest(BaseModel):
    graph_draft_id: UUID

    # defaults from your guide:
    connector_score_mode: Literal["degree", "pagerank", "none"] = "degree"
    layout_mode: Literal["none"] = "none"

    # bump this only when you change default-grouping logic
    grouping_policy_version: int = 1

class BuildMapResponse(BaseModel):
    map_id: UUID
    default_grouping: Grouping
    allowed_groupings: list[Grouping] = Field(default_factory=lambda: ["topic", "subfield"])
    node_count: int
    edge_count: int
    stats: dict[str, Any]

class FieldContextOut(BaseModel):
    field_id: str
    label: str
    node_count: int
    share: float

class GroupSummaryOut(BaseModel):
    group_id: str
    label: str
    node_count: int
    share: float

class MapNodeOut(BaseModel):
    work_id: str
    group_id: str

    # Work preview fields
    title: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = []
    venue: Optional[str] = None
    cited_by_count: Optional[int] = None

    # Access info
    access_status: Optional[Literal["open", "closed", "unknown"]] = None
    pdf_url: Optional[str] = None

    # Map fields
    connector_score: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None

class MapEdgeOut(BaseModel):
    from_work_id: str
    to_work_id: str

class MapRenderResponse(BaseModel):
    map_id: UUID
    field_contexts: list[FieldContextOut]

    active_grouping: Literal["topic", "subfield"]
    allowed_groupings: list[str]

    group_summaries: list[GroupSummaryOut]
    nodes: list[MapNodeOut]
    edges: list[MapEdgeOut]

class RankFilters(BaseModel):
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    topic_id: Optional[str] = None  # Strict filter by OpenAlex topic (for "Research this topic")

class RankParams(BaseModel):
    top_k: int = 100
    max_candidates_scored: int = 5000
    recency_half_life_years: float = 5.0

    w_rel: float = 0.60
    w_imp: float = 0.30
    w_rec: float = 0.08
    w_comp: float = 0.02

class DirectRankRequest(BaseModel):
    candidate_set_id: UUID
    context: Optional[dict[str, Any]] = None
    filters: Optional[RankFilters] = None
    params: Optional[RankParams] = None

class RankedItem(BaseModel):
    rank_index: int
    work_id: str
    score: float
    reasons: list[str]
    score_breakdown: dict[str, Any]
    preview: dict[str, Any]
    provenance: list[Any]
    evaluation: Optional[str] = None
    scoring: Optional[dict[str, Any]] = None

class QueryClassificationResponse(BaseModel):
    type: str
    confidence: float
    query_specificity: Optional[str] = None  # "broad" or "specific"


# ============================================================================
# Subtopic Generation Schemas
# ============================================================================

class SubtopicSuggestion(BaseModel):
    """A subtopic cluster within a broad query's results."""
    subtopic_id: str  # Generated ID (e.g., "subtopic_1")
    label: str  # Human-readable label (e.g., "Transformers & Attention")
    description: str  # Brief description of the subtopic
    representative_work_ids: list[str] = []  # Top papers in this subtopic (preview only)
    paper_count: int = 0
    # Query to use when drilling down into this subtopic via /rank endpoint
    # Frontend should call POST /rank with {"query_text": drill_down_query}
    drill_down_query: str = ""


class SubtopicsResponse(BaseModel):
    """Response for subtopic generation endpoint."""
    rank_job_id: UUID
    query_specificity: str  # "broad"
    subtopics: list[SubtopicSuggestion] = []


# ============================================================================
# Temporal Map Schemas
# ============================================================================

class TemporalPaper(BaseModel):
    """A paper in a temporal era."""
    work_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    cited_by_count: Optional[int] = None
    is_milestone: bool = False  # High citation count relative to era
    rank_in_results: Optional[int] = None  # Position in ranked list


class TemporalEra(BaseModel):
    """A group of papers from the same era."""
    label: str  # "1990s", "2000s", etc.
    start_year: int
    end_year: int
    papers: list[TemporalPaper] = []
    milestone_count: int = 0
    is_breakthrough_era: bool = False  # Era with significant paradigm shift


class BreakthroughPaper(BaseModel):
    """A paper that represents a breakthrough in the field."""
    work_id: str
    title: Optional[str] = None
    cited_by_count: int = 0


class Citation(BaseModel):
    """A citation linking a specific claim to a paper."""
    claim: str  # The specific fact/statistic being cited
    paper_title: str  # Title of the paper that supports this claim


class BreakthroughAnalysis(BaseModel):
    """Analysis of paradigm shift year for a field."""
    breakthrough_year: Optional[int] = None
    from_era: Optional[str] = None  # Era before the breakthrough (e.g., "1990s")
    breakthrough_era: Optional[str] = None  # Era when breakthrough occurred (e.g., "2010s")
    breakthrough_papers: list[BreakthroughPaper] = []  # Papers that define the breakthrough
    pre_breakthrough_approach: Optional[str] = None
    post_breakthrough_approach: Optional[str] = None
    shift_description: Optional[str] = None  # Technical explanation (no inline citations)
    shift_citations: list[Citation] = []  # Citations for specific facts/stats in shift_description
    rejection_reason: Optional[str] = None  # Why no paradigm shift (no inline citations)
    rejection_citations: list[Citation] = []  # Citations for specific facts in rejection_reason
    confidence: float = 0.0


class EvolutionTrend(BaseModel):
    """Methodology evolution for a single era."""
    era: str  # "1990s", "2000s"
    dominant_approach: str  # e.g., "Statistical learning", "Deep neural networks"
    key_themes: list[str] = []  # e.g., ["feature engineering", "model selection"]
    representative_paper_id: Optional[str] = None


class TemporalMapAnalytics(BaseModel):
    """Breakthrough and evolution analysis for temporal map."""
    breakthrough: Optional[BreakthroughAnalysis] = None
    evolution: list[EvolutionTrend] = []


class TemporalMapResponse(BaseModel):
    """Response for temporal map endpoint."""
    rank_job_id: UUID
    scope: Literal["broad", "subtopic"]
    scope_label: str  # "Machine Learning" or "Transformers & Attention"
    topic_id: Optional[str] = None
    subtopic_id: Optional[str] = None
    eras: list[TemporalEra] = []
    analytics: Optional[TemporalMapAnalytics] = None


class ConvergenceInfo(BaseModel):
    """Discovery curve convergence metrics from wave-based LLM scoring."""
    total_predicted: float  # N: predicted total relevance in candidate pool
    found: float            # cumulative relevance actually scored
    completeness: float     # found / total_predicted (0-1)
    confidence: float       # R² goodness of fit (0-1)
    tau: float              # scale parameter
    k: float                # shape parameter (>1 = front-loaded discovery)
    papers_scored: int      # total papers sent to LLM
    waves_completed: int    # number of scoring waves executed


class DirectRankResponse(BaseModel):
    rank_job_id: UUID
    display_title: Optional[str] = None  # Research space header (query text or paper title)
    job: dict[str, Any]
    query_classification: Optional[QueryClassificationResponse] = None
    convergence: Optional[ConvergenceInfo] = None
    # Flat list of ranked papers (used by drill-down endpoint)
    # When present, categorized fields are empty
    items: list[RankedItem] = []
    # SIX CATEGORIES (matching rank_service.py implementation):
    foundational: list[RankedItem] = []  # Field-defining foundational papers
    methodology: list[RankedItem] = []  # Methods, tools, techniques
    reviews: list[RankedItem] = []  # Review papers and surveys
    applications: list[RankedItem] = []  # Clinical trials, case studies, implementations
    textbooks: list[RankedItem] = []  # Educational materials
    additional_relevant: list[RankedItem] = []  # Quality papers that passed LLM+RRF but missed citation thresholds