"""
Data Models Package

Organized by category:
- work: WorkRef, WorkRefThin, Author, IngestState
- topic: TopicRef, TopicQueryRef, TopicHierarchy
- candidate: CandidateSet, CandidateItem
- graph: GraphDraft, GraphStats, CitationEdge
- map: Map, MapNode, SubtopicDefinition, FieldContext, LayoutCoordinates
- ranking: RankedList, RankedItem, RankingContext, ScoreBreakdown
"""

# Work models
from app.models.work import (
    WorkRef,
    WorkRefThin,
    Author,
    IngestState,
    TopicRefSimple,
)

# Topic models
from app.models.topic import (
    TopicRef,
    TopicQueryRef,
    TopicHierarchy,
)

# Candidate models
from app.models.candidate import (
    CandidateSet,
    CandidateItem,
)

# Graph models
from app.models.graph import (
    GraphDraft,
    GraphStats,
    CitationEdge,
)

# Map models
from app.models.map import (
    Map,
    MapNode,
    LayoutCoordinates,
    SubtopicDefinition,
    FieldContext,
)

# Ranking models
from app.models.ranking import (
    RankedList,
    RankedItem,
    RankingContext,
    ScoreBreakdown,
)

__all__ = [
    # Work
    "WorkRef",
    "WorkRefThin",
    "Author",
    "IngestState",
    "TopicRefSimple",
    # Topic
    "TopicRef",
    "TopicQueryRef",
    "TopicHierarchy",
    # Candidate
    "CandidateSet",
    "CandidateItem",
    # Graph
    "GraphDraft",
    "GraphStats",
    "CitationEdge",
    # Map
    "Map",
    "MapNode",
    "LayoutCoordinates",
    "SubtopicDefinition",
    "FieldContext",
    # Ranking
    "RankedList",
    "RankedItem",
    "RankingContext",
    "ScoreBreakdown",
]
