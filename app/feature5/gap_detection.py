"""
Gap detection algorithms for Feature 5: Research Gap Analysis.

This module implements heuristic detectors for each gap type:
- Structural gaps (citation graph)
- Coverage gaps (ranked list)
- Temporal gaps (timeline data)
- Methodological gaps (methodology comparisons)
- Novelty gaps (novelty assessments)

Each detector outputs scored candidates that are then evaluated and
filtered by the LLM before synthesis.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature5.schemas import (
    CoverageGapCandidate,
    MethodologicalGapCandidate,
    NoveltyGapCandidate,
    StructuralGapCandidate,
    TemporalGapCandidate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Structural Gap Detection (Citation Graph)
# =============================================================================

def _compute_topic_similarity(
    cluster_a: List[Dict],
    cluster_b: List[Dict],
) -> float:
    """
    Compute topic similarity between two clusters using Jaccard similarity
    over ALL topic IDs from each paper's topics_json field.

    Each paper has multiple topic assignments (T-prefix and C-prefix IDs)
    from OpenAlex. We collect all high-confidence topics across all papers
    in each cluster and compute Jaccard over those sets.

    Returns a value between 0.0 and 1.0.
    """
    import json

    topics_a: Set[str] = set()
    topics_b: Set[str] = set()

    for node in cluster_a:
        # Use multi-topic data from topics_json (richer than single best_topic_id)
        topics_json = node.get("topics_json")
        if topics_json:
            if isinstance(topics_json, str):
                try:
                    topics_json = json.loads(topics_json)
                except (json.JSONDecodeError, TypeError):
                    topics_json = []
            if isinstance(topics_json, list):
                for t in topics_json:
                    if isinstance(t, dict) and t.get("score", 0) > 0.5:
                        topics_a.add(t.get("topic_id", ""))
        # Fallback: use best_topic_id and best_subfield_id
        if node.get("best_topic_id"):
            topics_a.add(node["best_topic_id"])
        if node.get("best_subfield_id"):
            topics_a.add(node["best_subfield_id"])
        # Also use primary_topic_id from works
        if node.get("primary_topic_id"):
            topics_a.add(node["primary_topic_id"])

    for node in cluster_b:
        topics_json = node.get("topics_json")
        if topics_json:
            if isinstance(topics_json, str):
                try:
                    topics_json = json.loads(topics_json)
                except (json.JSONDecodeError, TypeError):
                    topics_json = []
            if isinstance(topics_json, list):
                for t in topics_json:
                    if isinstance(t, dict) and t.get("score", 0) > 0.5:
                        topics_b.add(t.get("topic_id", ""))
        if node.get("best_topic_id"):
            topics_b.add(node["best_topic_id"])
        if node.get("best_subfield_id"):
            topics_b.add(node["best_subfield_id"])
        if node.get("primary_topic_id"):
            topics_b.add(node["primary_topic_id"])

    # Remove empty strings
    topics_a.discard("")
    topics_b.discard("")

    if not topics_a or not topics_b:
        return 0.0

    intersection = topics_a & topics_b
    union = topics_a | topics_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def detect_structural_gaps(
    conn: Connection,
    map_id: UUID,
    max_candidates: int = 10,
) -> List[StructuralGapCandidate]:
    """
    Detect structural gaps in citation graph.

    Uses community detection via topic clustering and analyzes
    inter-cluster citation density to find disconnected but related clusters.
    Uses Jaccard similarity over topic/subfield IDs across all cluster papers.
    """
    candidates = []

    # Get nodes with their topics, multi-topic data, and citation relationships
    nodes = conn.execute(
        text("""
            SELECT
                mn.work_id,
                mn.best_topic_id,
                mn.best_subfield_id,
                w.title,
                w.cited_by_count,
                w.year,
                w.topics_json,
                w.primary_topic_id
            FROM map_nodes mn
            JOIN works w ON w.work_id = mn.work_id
            WHERE mn.map_id = :map_id
        """),
        {"map_id": map_id},
    ).mappings().all()

    if not nodes:
        return []

    # Group nodes by topic (as proxy for clusters)
    topic_clusters: Dict[str, List[Dict]] = defaultdict(list)
    for node in nodes:
        topic_id = node["best_topic_id"] or node["best_subfield_id"] or "unknown"
        topic_clusters[topic_id].append(dict(node))

    # Get edges for citation density calculation
    edges = conn.execute(
        text("""
            SELECT from_work_id, to_work_id
            FROM map_edges
            WHERE map_id = :map_id
        """),
        {"map_id": map_id},
    ).mappings().all()

    # Build edge lookup
    edge_set: Set[Tuple[str, str]] = {
        (e["from_work_id"], e["to_work_id"]) for e in edges
    }

    # Get topic display names
    topic_ids = list(topic_clusters.keys())
    topic_names = _get_topic_names(conn, topic_ids)

    # Calculate inter-cluster citation density for each pair
    cluster_ids = list(topic_clusters.keys())
    for i, cluster_a_id in enumerate(cluster_ids):
        for cluster_b_id in cluster_ids[i + 1:]:
            cluster_a = topic_clusters[cluster_a_id]
            cluster_b = topic_clusters[cluster_b_id]

            if len(cluster_a) < 2 or len(cluster_b) < 2:
                continue

            # Calculate citation density between clusters
            cross_citations = 0
            for node_a in cluster_a:
                for node_b in cluster_b:
                    if (node_a["work_id"], node_b["work_id"]) in edge_set:
                        cross_citations += 1
                    if (node_b["work_id"], node_a["work_id"]) in edge_set:
                        cross_citations += 1

            max_possible = len(cluster_a) * len(cluster_b) * 2
            citation_density = cross_citations / max_possible if max_possible > 0 else 0

            # Semantic topic similarity using Jaccard over all papers' topics
            topic_similarity = _compute_topic_similarity(cluster_a, cluster_b)

            # Gap score: high similarity but low citation density
            # Threshold lowered to 0.10 to catch gaps where fine-grained
            # topics dilute Jaccard despite meaningful shared concepts
            if topic_similarity < 0.10:
                continue

            gap_score = topic_similarity * (1.0 - citation_density)

            if gap_score > 0.15:
                candidates.append(StructuralGapCandidate(
                    cluster_a_id=cluster_a_id,
                    cluster_b_id=cluster_b_id,
                    cluster_a_label=topic_names.get(cluster_a_id, cluster_a_id),
                    cluster_b_label=topic_names.get(cluster_b_id, cluster_b_id),
                    topic_similarity=topic_similarity,
                    citation_density=citation_density,
                    gap_score=gap_score,
                    representative_papers_a=[n["work_id"] for n in cluster_a[:3]],
                    representative_papers_b=[n["work_id"] for n in cluster_b[:3]],
                ))

    # Sort by gap score and return top candidates
    candidates.sort(key=lambda x: x.gap_score, reverse=True)
    return candidates[:max_candidates]


def _get_topic_names(conn: Connection, topic_ids: List[str]) -> Dict[str, str]:
    """
    Get display names for topic IDs.

    Handles both T-prefix (OpenAlex topics) and C-prefix (OpenAlex concepts)
    IDs. For C-prefix IDs not found directly, falls back to looking up the
    primary_topic_id of papers that have that concept as their best_topic_id.
    """
    if not topic_ids:
        return {}

    names: Dict[str, str] = {}

    # First try direct lookup in openalex_topics
    result = conn.execute(
        text("""
            SELECT topic_id, display_name
            FROM openalex_topics
            WHERE topic_id = ANY(:topic_ids)
        """),
        {"topic_ids": topic_ids},
    ).mappings().all()

    for r in result:
        if r["display_name"]:
            names[r["topic_id"]] = r["display_name"]

    # For IDs not found, try fallback: find primary_topic_id for papers
    # with this best_topic_id, then look up that topic's display name
    missing = [tid for tid in topic_ids if tid not in names]
    if missing:
        fallback = conn.execute(
            text("""
                SELECT DISTINCT mn.best_topic_id, w.primary_topic_id
                FROM map_nodes mn
                JOIN works w ON w.work_id = mn.work_id
                WHERE mn.best_topic_id = ANY(:missing_ids)
                AND w.primary_topic_id IS NOT NULL
            """),
            {"missing_ids": missing},
        ).mappings().all()

        # Map C-prefix -> T-prefix using most common primary_topic_id
        from collections import Counter
        c_to_t_counts: Dict[str, Counter] = defaultdict(Counter)
        for r in fallback:
            c_to_t_counts[r["best_topic_id"]][r["primary_topic_id"]] += 1

        c_to_t: Dict[str, str] = {}
        for c_id, counter in c_to_t_counts.items():
            c_to_t[c_id] = counter.most_common(1)[0][0]

        if c_to_t:
            t_ids = list(set(c_to_t.values()))
            t_names = conn.execute(
                text("""
                    SELECT topic_id, display_name
                    FROM openalex_topics
                    WHERE topic_id = ANY(:t_ids)
                """),
                {"t_ids": t_ids},
            ).mappings().all()

            t_name_map = {r["topic_id"]: r["display_name"] for r in t_names if r["display_name"]}

            for c_id, t_id in c_to_t.items():
                if t_id in t_name_map:
                    names[c_id] = t_name_map[t_id]

    return names


# =============================================================================
# Coverage Gap Detection (Ranked List)
# =============================================================================

def detect_coverage_gaps(
    conn: Connection,
    map_id: UUID,
    query_text: Optional[str] = None,
    max_candidates: int = 10,
) -> List[CoverageGapCandidate]:
    """
    Detect coverage gaps in ranked paper list.

    Uses z-score deviation from mean topic representation to identify
    underrepresented topics, rather than assuming uniform distribution.
    Topics with only 1 paper are treated as peripheral unless they have
    high citation impact.
    """
    candidates = []

    # Get papers with their topics and citation counts
    papers = conn.execute(
        text("""
            SELECT
                mn.work_id,
                mn.best_topic_id,
                mn.best_subfield_id,
                w.title,
                w.primary_topic_id,
                w.cited_by_count
            FROM map_nodes mn
            JOIN works w ON w.work_id = mn.work_id
            WHERE mn.map_id = :map_id
        """),
        {"map_id": map_id},
    ).mappings().all()

    if not papers:
        return []

    total_papers = len(papers)

    # Count papers per topic
    topic_counts: Dict[str, int] = defaultdict(int)
    topic_papers: Dict[str, List[str]] = defaultdict(list)
    topic_citations: Dict[str, int] = defaultdict(int)

    for paper in papers:
        topic_id = paper["best_topic_id"] or paper["primary_topic_id"] or "unknown"
        topic_counts[topic_id] += 1
        topic_papers[topic_id].append(paper["work_id"])
        topic_citations[topic_id] += paper.get("cited_by_count", 0) or 0

    if len(topic_counts) < 2:
        return []

    # Get topic names
    topic_names = _get_topic_names(conn, list(topic_counts.keys()))

    # Calculate mean and std deviation of topic counts
    counts = list(topic_counts.values())
    mean_count = sum(counts) / len(counts)
    variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
    std_dev = math.sqrt(variance) if variance > 0 else 1.0

    # Dynamic threshold: lower for small maps where z-scores compress
    gap_score_threshold = 0.20 if total_papers < 50 else 0.30

    # Find underrepresented topics using z-score
    for topic_id, count in topic_counts.items():
        coverage_ratio = count / total_papers if total_papers > 0 else 0

        # Z-score: how many std deviations below mean
        z_score = (mean_count - count) / std_dev if std_dev > 0 else 0

        # Skip topics that are at or above average
        if z_score <= 0:
            continue

        # Skip singleton topics with low citation impact (likely peripheral)
        if count == 1:
            avg_citations_per_topic = (
                sum(topic_citations.values()) / len(topic_citations)
                if topic_citations else 0
            )
            if topic_citations[topic_id] < avg_citations_per_topic * 0.5:
                continue

        # Gap score from z-score, capped at 1.0
        gap_score = min(z_score / 2.0, 1.0)

        if gap_score > gap_score_threshold:
            candidates.append(CoverageGapCandidate(
                topic=topic_names.get(topic_id, topic_id),
                expected_in_query=count > 1 or topic_citations[topic_id] > mean_count * 10,
                coverage_ratio=coverage_ratio,
                gap_score=gap_score,
                related_papers=topic_papers[topic_id][:5],
            ))

    # Sort by gap score
    candidates.sort(key=lambda x: x.gap_score, reverse=True)
    return candidates[:max_candidates]


# =============================================================================
# Temporal Gap Detection (Timeline Data)
# =============================================================================

def detect_temporal_gaps(
    conn: Connection,
    map_id: UUID,
    max_candidates: int = 10,
) -> List[TemporalGapCandidate]:
    """
    Detect temporal gaps in publication timeline.

    Uses a rolling window average instead of flat global average to account
    for natural field growth. Compares each period to its local context
    rather than the overall average.
    """
    candidates = []

    # Get papers with their years
    papers = conn.execute(
        text("""
            SELECT
                mn.work_id,
                w.year,
                w.cited_by_count
            FROM map_nodes mn
            JOIN works w ON w.work_id = mn.work_id
            WHERE mn.map_id = :map_id
            AND w.year IS NOT NULL
        """),
        {"map_id": map_id},
    ).mappings().all()

    if not papers:
        return []

    # Guard: skip temporal detection on sparse datasets where year
    # distribution is too thin for meaningful gap detection
    if len(papers) < 15:
        logger.info(
            f"Skipping temporal gap detection: only {len(papers)} papers "
            f"(need >= 15 for meaningful timeline analysis)"
        )
        return []

    # Group by year
    year_counts: Dict[int, int] = defaultdict(int)
    for paper in papers:
        if paper["year"]:
            year_counts[paper["year"]] += 1

    if not year_counts:
        return []

    years = sorted(year_counts.keys())
    min_year, max_year = years[0], years[-1]

    if max_year - min_year < 3:
        # Not enough timeline span for meaningful gap detection
        return []

    # Build rolling window average (5-year window centered on each year)
    WINDOW_SIZE = 5
    half_window = WINDOW_SIZE // 2

    def local_average(year: int) -> float:
        """Average publication rate in window around year."""
        window_counts = [
            year_counts.get(y, 0)
            for y in range(year - half_window, year + half_window + 1)
            if min_year <= y <= max_year
        ]
        return sum(window_counts) / len(window_counts) if window_counts else 0

    # Find gaps using local context comparison
    current_gap_start = None
    gap_years: List[Tuple[int, int]] = []

    for year in range(min_year, max_year + 1):
        count = year_counts.get(year, 0)
        local_avg = local_average(year)

        # A year is "low" if it's below 40% of local rolling average
        # and the local average itself is meaningful (> 0.5 papers/year)
        is_low = count < local_avg * 0.4 and local_avg > 0.5

        if is_low:
            if current_gap_start is None:
                current_gap_start = year
        else:
            if current_gap_start is not None:
                gap_years.append((current_gap_start, year - 1))
                current_gap_start = None

    # Handle gap extending to present
    if current_gap_start is not None:
        gap_years.append((current_gap_start, max_year))

    # Global average for reference
    total_years = max_year - min_year + 1
    global_avg = len(papers) / total_years if total_years > 0 else 0

    # Create candidates for significant gaps (2+ years)
    for start_year, end_year in gap_years:
        if end_year - start_year >= 1:
            # Calculate gap metrics using local context
            period_rate = sum(
                year_counts.get(y, 0) for y in range(start_year, end_year + 1)
            ) / (end_year - start_year + 1)

            # Local average around the gap period
            context_avg = local_average((start_year + end_year) // 2)
            reference_avg = max(context_avg, global_avg * 0.5)

            raw_gap_score = (
                (reference_avg - period_rate) / reference_avg
                if reference_avg > 0 else 0
            )

            # Discount by dataset density: a gap in sparse data is less meaningful
            # At 2 papers/year, full score. At 0.5 papers/year, halved.
            density = len(papers) / total_years if total_years > 0 else 0
            density_discount = min(density / 2.0, 1.0)
            gap_score = raw_gap_score * density_discount

            # Check for comeback pattern
            has_activity_before = any(
                year_counts.get(y, 0) > local_average(y) * 0.5
                for y in range(start_year - 3, start_year)
                if y >= min_year
            )
            has_activity_after = any(
                year_counts.get(y, 0) > local_average(y) * 0.5
                for y in range(end_year + 1, end_year + 4)
                if y <= max_year
            )
            is_comeback = has_activity_before and has_activity_after

            if gap_score > 0.3:
                candidates.append(TemporalGapCandidate(
                    start_year=start_year,
                    end_year=end_year,
                    period_label=f"{start_year}-{end_year}",
                    publication_rate=period_rate,
                    average_rate=reference_avg,
                    gap_score=gap_score,
                    is_comeback=is_comeback,
                ))

    candidates.sort(key=lambda x: x.gap_score, reverse=True)
    return candidates[:max_candidates]


# =============================================================================
# Methodological Gap Detection
# =============================================================================

# Method categories for normalizing free-text approach descriptions.
# Order matters: first match wins, so more specific patterns come first.
_METHOD_CATEGORIES = [
    ("reinforcement learning", ["reinforcement learning", "rl ", "q-learning", "policy gradient", "actor-critic", "dqn", "ppo", "sac", "ddpg"]),
    ("deep learning", ["deep learning", "deep neural", "cnn", "convolutional neural", "resnet", "transformer", "attention mechanism", "bert", "gpt"]),
    ("imitation learning", ["imitation learning", "learning from demonstration", "behavioral cloning", "inverse reinforcement"]),
    ("transfer learning", ["transfer learning", "domain adaptation", "fine-tun", "pre-train"]),
    ("supervised learning", ["supervised learning", "classification", "regression", "labeled data"]),
    ("unsupervised learning", ["unsupervised learning", "clustering", "autoencoder", "self-supervised"]),
    ("simulation", ["simulat", "sim-to-real", "virtual environment", "physics engine"]),
    ("optimization", ["optimization", "evolutionary", "genetic algorithm", "bayesian optim", "gradient descent"]),
    ("control theory", ["control theory", "pid ", "model predictive control", "mpc", "feedback control", "optimal control"]),
    ("computer vision", ["computer vision", "image processing", "object detection", "visual", "visuomotor"]),
    ("natural language processing", ["natural language", "nlp", "text mining", "language model"]),
    ("survey", ["survey", "review", "systematic review", "meta-analysis", "literature review"]),
    ("analytical", ["analytical", "theoretical", "mathematical model", "formal analysis", "proof"]),
]


def _categorize_method(raw_method: str) -> str:
    """
    Categorize a free-text method description into a high-level category.

    Uses keyword matching against predefined categories. Returns "other"
    if no category matches.
    """
    if not raw_method or raw_method == "unknown":
        return "unknown"

    text = raw_method.lower()
    for category, keywords in _METHOD_CATEGORIES:
        for kw in keywords:
            if kw in text:
                return category

    return "other"


def detect_methodological_gaps(
    conn: Connection,
    map_id: UUID,
    max_candidates: int = 10,
) -> List[MethodologicalGapCandidate]:
    """
    Detect methodological gaps from cached methodology comparisons.

    Builds method x domain matrix and finds empty/sparse cells.
    Filters out nonsensical combinations by requiring both the method
    and domain to have sufficient representation (>= 2 papers each)
    and prioritizing combinations where BOTH neighbors are populated.
    """
    candidates = []

    # Get methodology fingerprints for papers in this map
    fingerprints = conn.execute(
        text("""
            SELECT
                mfc.work_id,
                mfc.fingerprint_json
            FROM methodology_fingerprint_cache mfc
            WHERE EXISTS (
                SELECT 1 FROM map_nodes mn
                WHERE mn.map_id = :map_id
                AND mn.work_id = mfc.work_id
            )
        """),
        {"map_id": map_id},
    ).mappings().all()

    if len(fingerprints) < 3:
        # Not enough methodology data
        return []

    # Extract methods and domains from fingerprints
    method_papers: Dict[str, List[str]] = defaultdict(list)
    domain_papers: Dict[str, List[str]] = defaultdict(list)
    method_domain_papers: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for fp in fingerprints:
        work_id = fp["work_id"]
        data = fp["fingerprint_json"] or {}

        # Extract method — fingerprint 'approach' is often free-text prose.
        # Normalize to high-level categories via keyword matching.
        raw_method = data.get("approach") or data.get("methodology_type") or "unknown"
        if isinstance(raw_method, list):
            raw_method = " ".join(str(x) for x in raw_method)
        raw_method = str(raw_method).lower().strip()
        method = _categorize_method(raw_method)

        # Extract domain (from domain or application area)
        domain = data.get("domain") or data.get("application_area") or "unknown"
        if isinstance(domain, list):
            domain = domain[0] if domain else "unknown"
        domain = str(domain).lower().strip()

        method_papers[method].append(work_id)
        domain_papers[domain].append(work_id)
        method_domain_papers[(method, domain)].append(work_id)

    # Filter to methods and domains with sufficient representation
    # This removes rare/noise entries that produce nonsensical combinations
    MIN_PAPERS_PER_AXIS = 2
    all_methods = {m for m in method_papers if m != "unknown" and len(method_papers[m]) >= MIN_PAPERS_PER_AXIS}
    all_domains = {d for d in domain_papers if d != "unknown" and len(domain_papers[d]) >= MIN_PAPERS_PER_AXIS}

    if len(all_methods) < 2 or len(all_domains) < 2:
        return []

    for method in all_methods:
        for domain in all_domains:
            cell_count = len(method_domain_papers.get((method, domain), []))

            if cell_count == 0:
                # Calculate directional neighbor density:
                # How many papers use this method in OTHER domains?
                method_in_other_domains = sum(
                    len(method_domain_papers.get((method, d), []))
                    for d in all_domains if d != domain
                )
                # How many papers in this domain use OTHER methods?
                domain_with_other_methods = sum(
                    len(method_domain_papers.get((m, domain), []))
                    for m in all_methods if m != method
                )

                # BOTH axes must be populated for this to be a meaningful gap.
                # If the method is never used outside its home domain, or the
                # domain only uses one method, the combination is likely noise.
                if method_in_other_domains < 1 or domain_with_other_methods < 1:
                    continue

                # Score based on geometric mean of both axes
                avg_neighbor = math.sqrt(method_in_other_domains * domain_with_other_methods)
                gap_score = min(avg_neighbor / 5, 1.0)

                if gap_score > 0.3:
                    candidates.append(MethodologicalGapCandidate(
                        method=method,
                        domain=domain,
                        neighbor_density=avg_neighbor,
                        gap_score=gap_score,
                        related_method_papers=method_papers[method][:3],
                        related_domain_papers=domain_papers[domain][:3],
                    ))

    candidates.sort(key=lambda x: x.gap_score, reverse=True)
    return candidates[:max_candidates]


# =============================================================================
# Novelty Gap Detection
# =============================================================================

def detect_novelty_gaps(
    conn: Connection,
    map_id: UUID,
    max_candidates: int = 10,
) -> List[NoveltyGapCandidate]:
    """
    Detect novelty gaps from cached novelty assessments.

    Identifies ACTIVE clusters (recent publications) with consistently low novelty.
    A cluster of only old papers with incremental work is likely mature/solved.
    A cluster with recent papers but no innovation signals genuine opportunity.
    """
    candidates = []

    # Get novelty assessments AND publication years for recency filtering
    assessments = conn.execute(
        text("""
            SELECT
                mn.work_id,
                mn.best_topic_id,
                mn.best_subfield_id,
                ndc.novelty_assessment,
                w.year
            FROM map_nodes mn
            JOIN node_details_cache ndc ON ndc.work_id = mn.work_id
            JOIN works w ON w.work_id = mn.work_id
            WHERE mn.map_id = :map_id
            AND ndc.novelty_assessment IS NOT NULL
        """),
        {"map_id": map_id},
    ).mappings().all()

    if len(assessments) < 3:
        return []

    # Map novelty tiers to scores
    NOVELTY_SCORES = {
        "LOW": 0.25,
        "MEDIUM": 0.50,
        "HIGH": 0.75,
        "PIONEERING": 1.0,
    }

    # Group by topic/cluster
    cluster_data: Dict[str, List[Dict]] = defaultdict(list)

    for a in assessments:
        cluster_id = a["best_topic_id"] or a["best_subfield_id"] or "unknown"
        assessment = a["novelty_assessment"] or {}
        tier = assessment.get("tier", "MEDIUM")
        score = NOVELTY_SCORES.get(tier, 0.5)
        cluster_data[cluster_id].append({
            "work_id": a["work_id"],
            "score": score,
            "year": a.get("year"),
        })

    # Get topic names
    topic_names = _get_topic_names(conn, list(cluster_data.keys()))

    # Current year for recency check
    import datetime
    current_year = datetime.datetime.now().year

    # Find ACTIVE clusters lacking pioneering work
    for cluster_id, data in cluster_data.items():
        if len(data) < 3:
            continue

        scores = [d["score"] for d in data]
        avg_novelty = sum(scores) / len(scores)
        max_novelty = max(scores)

        # Require the cluster to be active: at least one paper from
        # the last 5 years. Otherwise it's likely a mature/completed area.
        recent_papers = [d for d in data if d.get("year") and d["year"] >= current_year - 5]
        if not recent_papers:
            continue

        # Require consistently low novelty (not just absence of pioneering)
        # Average must be below MEDIUM and no paper reaches HIGH
        if avg_novelty >= 0.50 or max_novelty >= 0.75:
            continue

        # Gap score: combines low novelty ceiling with cluster activity
        # Active clusters with uniformly low novelty are the strongest signals
        activity_ratio = len(recent_papers) / len(data)
        gap_score = (1.0 - max_novelty) * (0.5 + 0.5 * activity_ratio)

        if gap_score > 0.25:
            candidates.append(NoveltyGapCandidate(
                cluster_id=cluster_id,
                cluster_label=topic_names.get(cluster_id, cluster_id),
                avg_novelty=avg_novelty,
                max_novelty=max_novelty,
                gap_score=gap_score,
                papers_in_cluster=[d["work_id"] for d in data],
            ))

    candidates.sort(key=lambda x: x.gap_score, reverse=True)
    return candidates[:max_candidates]


# =============================================================================
# Main Detection Orchestrator
# =============================================================================

def detect_all_gaps(
    conn: Connection,
    map_id: UUID,
    available_sources: List[str],
) -> Dict[str, List[Any]]:
    """
    Run all applicable gap detectors based on available data sources.

    Args:
        conn: Database connection
        map_id: Map ID to analyze
        available_sources: List of available data sources

    Returns:
        Dictionary mapping gap type to list of candidates
    """
    results: Dict[str, List[Any]] = {}

    # Always run structural detection if we have citation graph
    if "citation_graph" in available_sources:
        logger.info(f"Detecting structural gaps for map {map_id}")
        results["structural"] = detect_structural_gaps(conn, map_id)

    # Always run coverage detection (works with any entry point)
    logger.info(f"Detecting coverage gaps for map {map_id}")
    results["coverage"] = detect_coverage_gaps(conn, map_id)

    # Run temporal detection if timeline data available
    if "timeline" in available_sources:
        logger.info(f"Detecting temporal gaps for map {map_id}")
        results["temporal"] = detect_temporal_gaps(conn, map_id)

    # Run methodological detection if methodology data available
    if "methodology" in available_sources:
        logger.info(f"Detecting methodological gaps for map {map_id}")
        results["methodological"] = detect_methodological_gaps(conn, map_id)

    # Run novelty detection if novelty data available
    if "novelty" in available_sources:
        logger.info(f"Detecting novelty gaps for map {map_id}")
        results["novelty"] = detect_novelty_gaps(conn, map_id)

    # Log summary
    total = sum(len(v) for v in results.values())
    logger.info(f"Detected {total} total gap candidates across {len(results)} types")

    return results
