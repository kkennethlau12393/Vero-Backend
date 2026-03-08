"""
Candidate scoring for citation map v2.

Scores papers on three dimensions:
1. Relevance — how well the paper matches the structured query
2. Connectivity — how many edges it shares with papers already in the graph
3. Temporal — bias toward seminal or recent papers

All candidates across all hops are scored the same way.
The graph is built greedily, adding the best-scoring candidate at each step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def compute_relevance_score(
    paper: Dict[str, Any],
    structured_query: Dict[str, Any],
    scope: str,
) -> float:
    """Score paper relevance against the structured query.

    Uses title + abstract keyword matching against topic, domain, aspect
    and their aliases. Score is 0.0 to 1.0.
    """
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text = f"{title} {abstract}"

    topic = (structured_query.get("topic") or "").lower()
    domain = (structured_query.get("domain") or "").lower()
    aspect = (structured_query.get("aspect") or "").lower()

    topic_terms = [topic] + [a.lower() for a in structured_query.get("topic_aliases", [])]
    domain_terms = [domain] + [a.lower() for a in structured_query.get("domain_aliases", [])]
    aspect_terms = [aspect] + [a.lower() for a in structured_query.get("aspect_aliases", [])]

    # Remove empty strings
    topic_terms = [t for t in topic_terms if t]
    domain_terms = [t for t in domain_terms if t]
    aspect_terms = [t for t in aspect_terms if t]

    topic_match = any(t in text for t in topic_terms) if topic_terms else False
    domain_match = any(t in text for t in domain_terms) if domain_terms else False
    aspect_match = any(t in text for t in aspect_terms) if aspect_terms else False

    # Score based on scope
    if scope == "intersection":
        if topic_match and domain_match:
            base = 0.8
        elif topic_match or domain_match:
            base = 0.4
        else:
            base = 0.1
    elif scope == "topic_focused":
        if topic_match:
            base = 0.8
        elif domain_match:
            base = 0.3
        else:
            base = 0.1
    elif scope == "domain_focused":
        if domain_match:
            base = 0.8
        elif topic_match:
            base = 0.3
        else:
            base = 0.1
    else:  # broad
        if topic_match or domain_match:
            base = 0.7
        else:
            base = 0.1

    # Aspect bonus
    if aspect_match:
        base = min(1.0, base + 0.15)

    # Title match bonus (title is more signal-dense than abstract)
    title_topic = any(t in title for t in topic_terms) if topic_terms else False
    title_domain = any(t in title for t in domain_terms) if domain_terms else False
    if title_topic and title_domain:
        base = min(1.0, base + 0.1)

    return base


def compute_temporal_score(
    paper: Dict[str, Any],
    temporal: str,
    current_year: int = 2026,
) -> float:
    """Score paper based on temporal preference.

    - seminal: favor older, highly-cited papers
    - recent: favor papers from last 5 years
    - all: neutral (0.5)
    """
    year = paper.get("year") or current_year
    cited = paper.get("cited_by_count") or 0

    if temporal == "seminal":
        age = max(0, current_year - year)
        age_score = min(1.0, age / 20)  # peaks at 20+ years old
        cite_score = min(1.0, cited / 1000)  # peaks at 1000+ citations
        return 0.4 * age_score + 0.6 * cite_score

    elif temporal == "recent":
        age = max(0, current_year - year)
        if age <= 2:
            return 1.0
        elif age <= 5:
            return 0.8
        elif age <= 10:
            return 0.4
        else:
            return 0.1

    else:  # all
        return 0.5


def compute_connectivity_score(
    paper_id: str,
    graph_ids: Set[str],
    edges: Dict[str, Set[str]],
) -> float:
    """Score how well-connected a paper is to the existing graph.

    Returns 0.0 for isolated papers, up to 1.0 for highly connected ones.
    """
    if not graph_ids:
        return 0.5  # first papers get neutral score

    connections = edges.get(paper_id, set())
    shared = connections & graph_ids
    num_shared = len(shared)

    if num_shared == 0:
        return 0.0
    elif num_shared == 1:
        return 0.3
    elif num_shared <= 3:
        return 0.6
    elif num_shared <= 6:
        return 0.8
    else:
        return 1.0


def get_drift_thresholds(drift: str) -> Dict[str, float]:
    """Get relevance thresholds per hop level based on drift preference.

    Papers below the threshold for their hop level are discarded
    before scoring.
    """
    if drift == "strict":
        return {
            "hop1_threshold": 0.5,
            "hop2_threshold": 0.45,
            "hop3_threshold": 0.4,
            "max_hops": 2,
        }
    elif drift == "moderate":
        return {
            "hop1_threshold": 0.4,
            "hop2_threshold": 0.3,
            "hop3_threshold": 0.2,
            "max_hops": 2,
        }
    else:  # open
        return {
            "hop1_threshold": 0.3,
            "hop2_threshold": 0.2,
            "hop3_threshold": 0.1,
            "max_hops": 3,
        }


def get_fetch_limits(map_size: str) -> Dict[str, int]:
    """Get fetch limits per hop based on map size.

    Larger maps need more candidates to select from.
    """
    if map_size == "small":
        return {
            "target_nodes": 25,
            "hop1_fetch": 50,
            "hop2_fetch": 20,
            "hop2_expand_count": 5,
        }
    elif map_size == "large":
        return {
            "target_nodes": 60,
            "hop1_fetch": 120,
            "hop2_fetch": 40,
            "hop2_expand_count": 12,
        }
    else:  # medium
        return {
            "target_nodes": 40,
            "hop1_fetch": 80,
            "hop2_fetch": 30,
            "hop2_expand_count": 8,
        }


def greedy_graph_select(
    candidates: List[Dict[str, Any]],
    edges: Dict[str, Set[str]],
    seed_id: str,
    target_size: int,
    structured_query: Dict[str, Any],
    scope: str,
    temporal: str,
    min_connectivity: float = 0.0,
    backbone_ratio: float = 0.6,
    relevance_floor: float = 0.0,
) -> List[str]:
    """Greedily select papers to build a connected, relevant graph.

    Uses two-phase selection:
    1. Graph backbone (backbone_ratio of target): only graph-traversal candidates
    2. Enrichment fill (remaining slots): all candidates including retrieval

    This ensures the citation graph structure is preserved while still
    incorporating relevant papers found via text retrieval.

    Parameters
    ----------
    candidates : list[dict]
        All candidate papers. Papers with _source="retrieval_enrichment"
        are deferred to phase 2.
    edges : dict[str, set[str]]
        Full adjacency map.
    seed_id : str
        Work ID of the seed paper.
    target_size : int
        Total nodes to select (including seed).
    structured_query : dict
        Topic/domain/aspect + aliases.
    scope : str
        Relevance scope.
    temporal : str
        Temporal preference.
    min_connectivity : float
        Minimum connectivity score for inclusion.
    backbone_ratio : float
        Fraction of target_size reserved for graph-traversal candidates (0.0-1.0).
        Default 0.6 means 60% graph backbone, 40% open to enrichment.
    relevance_floor : float
        Minimum relevance score for inclusion. Papers below this are skipped
        regardless of connectivity. Default 0.0 (no floor).

    Returns
    -------
    list[str]
        Ordered list of selected work_ids (seed first).
    """
    # Pre-compute relevance and temporal scores for all candidates
    scored = {}
    for c in candidates:
        wid = c["work_id"]
        if wid == seed_id:
            continue
        rel = compute_relevance_score(c, structured_query, scope)
        temp = compute_temporal_score(c, temporal)
        scored[wid] = {
            "relevance": rel,
            "temporal": temp,
            "paper": c,
            "is_enrichment": c.get("_source") == "retrieval_enrichment",
        }

    # Cap candidate pool: keep top 3x target_size by relevance + temporal
    if len(scored) > target_size * 3:
        top_candidates = sorted(
            scored.items(),
            key=lambda x: x[1]["relevance"] + x[1]["temporal"],
            reverse=True,
        )[:target_size * 3]
        scored = dict(top_candidates)

    selected = [seed_id]
    selected_set = {seed_id}

    def _select_best(eligible: Set[str]) -> str | None:
        """Pick the best candidate from eligible set."""
        best_id = None
        best_score = -1.0
        for wid in eligible:
            s = scored[wid]
            if s["relevance"] < relevance_floor:
                continue
            conn = compute_connectivity_score(wid, selected_set, edges)
            if conn < min_connectivity and len(eligible) > (target_size - len(selected)) * 2:
                continue
            combined = (
                0.45 * s["relevance"]
                + 0.35 * conn
                + 0.20 * s["temporal"]
            )
            if combined > best_score:
                best_score = combined
                best_id = wid
        return best_id

    # Split candidates into graph-traversal and enrichment
    graph_candidates = set(wid for wid, s in scored.items() if not s["is_enrichment"])
    all_remaining = set(scored.keys())

    # Phase 1: Graph backbone — select from graph-traversal candidates only
    backbone_target = max(1, int(target_size * backbone_ratio))
    while len(selected) < backbone_target and graph_candidates:
        best = _select_best(graph_candidates)
        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        graph_candidates.discard(best)
        all_remaining.discard(best)

    # Phase 2: Enrichment fill — select from ALL remaining candidates
    while len(selected) < target_size and all_remaining:
        best = _select_best(all_remaining)
        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        all_remaining.discard(best)
        graph_candidates.discard(best)

    return selected
