# Models Package Structure

All data models are organized in `app/models/` package by domain category.

## Quick Reference

```
app/models/
├── __init__.py          # Exports all 21 models (88 lines)
├── work.py              # Work/paper models (82 lines)
├── topic.py             # Topic/subject models (44 lines)
├── candidate.py         # Discovery results (40 lines)
├── graph.py             # Citation graph (55 lines)
├── map.py               # Finalized visualization (73 lines)
└── ranking.py           # Recommendations (60 lines)
                         # TOTAL: 442 lines
```

## Module Breakdown

### `work.py` (4 classes)
Work/paper representations and metadata.

```python
from app.models.work import (
    WorkRef,           # Full paper metadata
    WorkRefThin,       # Lightweight version
    Author,            # Author information
    IngestState,       # Data completeness enum
    TopicRefSimple     # Simple topic reference
)
```

**Key Features:**
- `WorkRef`: Complete paper with DOI, abstract, authors, citations, topics, OA URL
- `WorkRefThin`: Minimal fields for performance
- `IngestState`: RESOLVED, PARTIALLY_RESOLVED, UNRESOLVED
- Progressive enrichment support

### `topic.py` (3 classes)
Research topics and subject area classification.

```python
from app.models.topic import (
    TopicRef,          # Topic with hierarchy and score
    TopicQueryRef,     # Topics extracted from query
    TopicHierarchy     # Domain→Field→Subfield→Topic structure
)
```

**Key Features:**
- OpenAlex hierarchy support
- Relevance scoring
- Query-based topic extraction

### `candidate.py` (2 classes)
Discovery phase - pools of papers fetched from OpenAlex.

```python
from app.models.candidate import (
    CandidateSet,      # Pool of discovered papers
    CandidateItem      # Individual paper with metadata
)
```

**Key Features:**
- Immutable snapshots
- Seed type tracking (paper, topic_text, node_subtopic)
- Discovery parameter preservation
- Individual paper selection reasons

### `graph.py` (3 classes)
Temporary citation graph before finalization.

```python
from app.models.graph import (
    GraphDraft,        # Temporary graph (ephemeral, 24h TTL)
    GraphStats,        # Quality and coverage metrics
    CitationEdge       # Directed citation relationship
)
```

**Key Features:**
- Ephemeral (expires in 24 hours)
- Progressive enrichment
- Quality metrics (coverage %, resolved nodes, etc.)
- WorkRefThin nodes initially, can be enriched

### `map.py` (5 classes)
Finalized citation graph with visualization data.

```python
from app.models.map import (
    Map,                    # Finalized graph
    MapNode,               # Node with full metadata + position
    SubtopicDefinition,    # Node grouping strategy
    FieldContext,          # Research field context
    LayoutCoordinates      # 2D visualization position
)
```

**Key Features:**
- Permanent, versioned storage
- Full WorkRef nodes with complete metadata
- Layout coordinates for consistent visualization
- Intelligent subtopic grouping (auto-selects level)
- User override support

### `ranking.py` (4 classes)
Scored and ordered paper recommendations.

```python
from app.models.ranking import (
    RankedList,        # Scored paper recommendations
    RankedItem,        # Single ranked paper
    RankingContext,    # What ranking is based on
    ScoreBreakdown     # Transparent score calculation
)
```

**Key Features:**
- Transparent scoring with breakdowns
- Multiple scoring factors (topic relevance, citations, recency, similarity, centrality)
- Human-readable explanations
- Context-aware ranking (topic, subtopic, seed_paper)

## Import Patterns

### Pattern 1: Package-level import (recommended)
```python
from app.models import WorkRef, TopicRef, CandidateSet, IngestState

work = WorkRef(title="Paper", ingest_state=IngestState.RESOLVED)
```

### Pattern 2: Module-level import
```python
from app.models.work import WorkRef, IngestState
from app.models.topic import TopicRef

work = WorkRef(title="Paper", ingest_state=IngestState.RESOLVED)
```

### Pattern 3: Submodule import
```python
import app.models.work as work_models

work = work_models.WorkRef(...)
```

## Design Principles

1. **Progressive Enrichment**: Start with minimal data, enrich over time
2. **Immutability**: CandidateSet and Map are immutable snapshots
3. **Transparency**: RankedList includes detailed score breakdowns
4. **OpenAlex Compatible**: Uses standard identifiers (W..., T..., F..., A...)
5. **Quality Tracking**: IngestState and GraphStats monitor data completeness
6. **Versioning**: Maps are versioned for change tracking
7. **Expiration**: GraphDrafts are ephemeral with 24-hour TTL

## Model Relationships

```
WorkRef ← Primary domain object
  ├─ TopicRefSimple (within work)
  └─ Author[]

CandidateSet ← Discovery results
  ├─ CandidateItem[]
  └─ seeds: WorkRef or TopicQueryRef

GraphDraft ← Temporary graph
  ├─ nodes: WorkRefThin[]
  ├─ edges: CitationEdge[]
  ├─ stats: GraphStats
  └─ seed: WorkRefThin or TopicQueryRef

Map ← Finalized graph (versioned)
  ├─ nodes: MapNode[]
  │   └─ work: WorkRef
  │   └─ assigned_subtopic: TopicRef
  │   └─ position: LayoutCoordinates
  ├─ edges: CitationEdge[]
  ├─ field_context: FieldContext
  └─ subtopic_definition: SubtopicDefinition

RankedList ← Recommendations
  ├─ ranked_items: RankedItem[]
  │   └─ score_breakdown: ScoreBreakdown
  ├─ context: RankingContext
  └─ candidate_set_id: reference to CandidateSet
```

## Adding New Models

1. **Identify category**: Does it fit work, topic, candidate, graph, map, or ranking?
2. **Add to appropriate file**: Update that module with new class
3. **Update exports**: Add to `__init__.py` exports list
4. **Document**: Add docstrings and config examples
5. **Test**: Verify imports and instantiation

## See Also

- `ARCHITECTURE.md` - Overall architecture documentation
- `DATA_MODELS.md` - Complete model reference
- `AI_CONTEXT.md` - Context for AI assistants
