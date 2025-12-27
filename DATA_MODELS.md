# Data Models Reference

All data models are defined in `app/models.py` as Pydantic models. This document provides a quick reference for all available models.

## Import Statement

```python
from app.models import (
    WorkRef, WorkRefThin, Author, IngestState,
    TopicRef, TopicQueryRef, TopicHierarchy,
    CandidateSet, CandidateItem,
    GraphDraft, GraphStats, CitationEdge,
    Map, MapNode, SubtopicDefinition, FieldContext, LayoutCoordinates,
    RankedList, RankedItem, RankingContext, ScoreBreakdown
)
```

## Model Categories

### 1. Work Models - Academic Papers

#### WorkRef
Complete representation of a research paper. Canonical structure regardless of input source.

**Key Fields:**
- `work_id` (str): OpenAlex ID (W...)
- `doi` (str): Digital Object Identifier
- `title`, `abstract`, `year`, `venue`
- `authors` (List[Author])
- `cited_by_count` (int)
- `references` (List[str]): Work IDs cited by this paper
- `primary_topic` (TopicRefSimple)
- `topics` (List[TopicRefSimple])
- `oa_pdf_url` (HttpUrl): Open access PDF
- `ingest_state` (IngestState): Data completeness

**Usage:**
```python
work = WorkRef(
    work_id="W2741809807",
    title="Paper Title",
    ingest_state=IngestState.RESOLVED
)
```

#### WorkRefThin
Lightweight version for performance-critical operations.

**Fields:** `work_id`, `title`, `year`, `cited_by_count`, `ingest_state`

#### Author
Author information within a work.

**Fields:** `name`, `author_id`, `orcid`, `position`

#### IngestState (Enum)
Data completeness status: `RESOLVED`, `PARTIALLY_RESOLVED`, `UNRESOLVED`

---

### 2. Topic Models - Research Areas

#### TopicRef
Reference to a research topic with OpenAlex hierarchy.

**Key Fields:**
- `topic_id` (str): OpenAlex ID (T...)
- `display_name` (str): Human-readable name
- `score` (float): Relevance score (0-1)
- `hierarchy` (TopicHierarchy): Domain → Field → Subfield → Topic

**Score Interpretation:**
- In `Work.topics`: Strength of paper-topic association
- In query analysis: Strength of query-topic association

#### TopicQueryRef
Collection of topics extracted from user text query.

**Fields:**
- `query_text` (str)
- `topics` (List[TopicRef])
- `analyzed_at` (str): ISO timestamp

#### TopicHierarchy
OpenAlex hierarchy structure.

**Fields:** `domain_id/name`, `field_id/name`, `subfield_id/name`

---

### 3. Candidate Models - Discovery Results

#### CandidateSet
Pool of papers fetched from OpenAlex during discovery (Feature 1).

**Key Fields:**
- `candidate_set_id` (str): Unique identifier
- `seed_type` (Literal): "paper" | "topic_text" | "node_subtopic"
- `seed_ref` (str): work_id, topic_id, or query hash
- `field_context_id` (str): OpenAlex field ID
- `params_json` (dict): Discovery parameters
- `items` (List[CandidateItem])
- `created_at`, `total_count`

**Usage:**
```python
candidate_set = CandidateSet(
    seed_type="paper",
    seed_ref="W123",
    items=[CandidateItem(work_id="W456", reason="cited_by_seed")]
)
```

#### CandidateItem
Individual paper in candidate set.

**Fields:** `work_id`, `reason`, `reason_detail`, `fetched_at`, `relevance_score`

---

### 4. Graph Models - Citation Networks

#### GraphDraft
Temporary citation graph before finalization. **Ephemeral** (expires in 24h).

**Key Fields:**
- `graph_draft_id` (str)
- `seed` (Union[WorkRefThin, TopicQueryRef])
- `candidate_set_id` (str)
- `nodes` (List[WorkRefThin])
- `edges` (List[CitationEdge])
- `stats` (GraphStats)
- `created_at`, `updated_at`, `expires_at`

**Lifecycle:**
1. Create draft from CandidateSet
2. Enrich progressively (abstracts, topics)
3. User reviews stats
4. Finalize → Map
5. Expires after 24h

#### CitationEdge
Directed citation relationship.

**Fields:** `from_work_id`, `to_work_id`, `citation_context`

#### GraphStats
Quality and coverage metrics.

**Fields:** `node_count`, `edge_count`, `nodes_with_abstract`, `nodes_with_topics`, `abstract_coverage`, `topic_coverage`, `avg_citations_per_node`

---

### 5. Map Models - Finalized Visualizations

#### Map
Permanent stored citation graph with visualization data. **Versioned**.

**Key Fields:**
- `map_id` (str)
- `graph_draft_id` (str): Source draft
- `nodes` (List[MapNode]): Full metadata + positions
- `edges` (List[CitationEdge])
- `field_context` (FieldContext)
- `subtopic_definition` (SubtopicDefinition)
- `created_at`, `version`
- `title`, `description`, `is_public`

**Usage:**
```python
map_obj = Map(
    graph_draft_id="gd_123",
    nodes=[MapNode(...)],
    edges=[CitationEdge(...)],
    field_context=FieldContext(...),
    subtopic_definition=SubtopicDefinition(...)
)
```

#### MapNode
Full node with visualization data.

**Fields:** `work` (WorkRef), `assigned_subtopic`, `position`, `is_seed`, `importance_score`

#### SubtopicDefinition
Node grouping strategy configuration.

**Fields:**
- `level` (Literal): "domain" | "field" | "subfield" | "topic"
- `auto_selected` (bool)
- `reason` (str): Why this level chosen
- `user_override` (bool)

**Rules:**
- Default: subfield level
- < 4 subfields: topic level
- One subfield > 80% nodes: topic level

#### FieldContext
Research field context.

**Fields:** `field_id`, `field_name`, `subfield_count`

#### LayoutCoordinates
2D visualization position.

**Fields:** `x`, `y`

---

### 6. Ranking Models - Recommendations

#### RankedList
Scored and ordered paper recommendations.

**Key Fields:**
- `rank_id` (str)
- `context` (RankingContext): What ranking based on
- `candidate_set_id` (str)
- `ranked_items` (List[RankedItem])
- `created_at`, `algorithm`, `total_candidates`

**Use Cases:**
- Topic ranking: "Papers about ML"
- Similarity: "Papers like this one"
- Subtopic: "Papers in category"
- "Read more": Explore subtopic

#### RankedItem
Single ranked paper with explanation.

**Fields:** `work_id`, `score`, `score_breakdown`, `reasons`, `rank_position`

#### RankingContext
Defines ranking basis.

**Fields:**
- `context_type` (Literal): "topic" | "subtopic" | "seed_paper"
- `topic_id`, `subtopic_id`, `seed_work_id`
- `filters` (dict)

#### ScoreBreakdown
Transparent score calculation.

**Fields:** `total_score`, `topic_relevance`, `citation_count`, `recency`, `similarity`, `graph_centrality`, `custom_factors`

---

## Design Principles

1. **Progressive Enrichment**: Models support starting with minimal data and enriching over time
2. **Immutability**: CandidateSet and Map are immutable snapshots
3. **Transparency**: RankedList includes score breakdowns and human-readable reasons
4. **OpenAlex Compatible**: Uses standard OpenAlex identifiers (W..., T..., F..., A...)
5. **Quality Tracking**: IngestState and GraphStats track data completeness
6. **Versioning**: Maps are versioned for change tracking
7. **Expiration**: GraphDrafts are ephemeral (24h TTL)

## Common Patterns

### Create Work from Minimal Data
```python
work = WorkRef(
    title="Unknown Paper",
    ingest_state=IngestState.UNRESOLVED
)
# Enrich later when data available
work.work_id = "W123"
work.abstract = "Full abstract..."
work.ingest_state = IngestState.RESOLVED
```

### Use in API Response
```python
from app.models import WorkRef

@router.get("/work/{work_id}", response_model=WorkRef)
async def get_work(work_id: str) -> WorkRef:
    # Fetch and return work
    return WorkRef(...)
```

### Filter by Ingest State
```python
from app.models import IngestState

resolved_works = [w for w in works if w.ingest_state == IngestState.RESOLVED]
```

### Access Topic Hierarchy
```python
topic = work.primary_topic
if topic and topic.hierarchy:
    print(f"Field: {topic.hierarchy.field_name}")
    print(f"Subfield: {topic.hierarchy.subfield_name}")
```

## See Also

- `ARCHITECTURE.md` - Detailed architecture documentation
- `AI_CONTEXT.md` - Context for AI assistants
- `app/models.py` - Source code with full docstrings
