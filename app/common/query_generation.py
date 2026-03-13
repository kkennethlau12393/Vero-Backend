"""
Multi-query generation for structured query retrieval.

Generates multiple targeted retrieval queries based on the decomposed
query structure and user's scope/focus preferences.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def generate_retrieval_queries(structured: Dict[str, Optional[str]], scope: str) -> List[str]:
    """Generate multiple retrieval queries based on decomposition and scope.

    Uses topic_aliases and domain_aliases to build a rich set of queries,
    similar to the concept-based expansion path.

    Parameters
    ----------
    structured : dict
        Decomposed query with keys: topic, domain, aspect,
        topic_aliases, domain_aliases.
    scope : str
        One of: intersection, broad, topic_focused, domain_focused.

    Returns
    -------
    list[str]
        List of query strings for retrieval APIs.
    """
    topic = structured.get("topic", "")
    domain = structured.get("domain")
    aspect = structured.get("aspect")
    topic_aliases = structured.get("topic_aliases", []) or []
    domain_aliases = structured.get("domain_aliases", []) or []

    # Build full lists of topic and domain terms
    topic_terms = [topic] + [a for a in topic_aliases if a.lower() != topic.lower()]
    domain_terms = ([domain] + [a for a in domain_aliases if a.lower() != (domain or "").lower()]) if domain else []

    queries: List[str] = []
    seen = set()

    def add(q: str):
        q = q.strip()
        key = q.lower()
        if key and key not in seen:
            seen.add(key)
            queries.append(q)

    if scope == "intersection" and domain:
        # Cross-product: each topic term × each domain term
        for t in topic_terms:
            for d in domain_terms:
                add(f"{t} {d}")
        # Phrased variants for top terms
        add(f"{domain} applications of {topic}")
        add(f"{topic} applied to {domain}")
        add(f"impact of {domain} on {topic}")
        add(f"role of {domain} in {topic}")
        if aspect:
            add(f"{aspect} {topic} {domain}")
        # Individual terms as fallback (lower priority but ensures coverage)
        for t in topic_terms[:3]:
            add(t)
        for d in domain_terms[:3]:
            add(d)

    elif scope == "broad" and domain:
        # Cross-product of top terms
        for t in topic_terms[:3]:
            for d in domain_terms[:3]:
                add(f"{t} {d}")
        # Individual terms
        for t in topic_terms:
            add(t)
        for d in domain_terms:
            add(d)
        add(f"{domain} computational methods")

    elif scope == "topic_focused":
        for t in topic_terms:
            add(t)
        add(f"recent advances in {topic}")
        add(f"survey of {topic}")
        # Light domain cross for context
        if domain:
            for t in topic_terms[:2]:
                add(f"{t} {domain}")

    elif scope == "domain_focused" and domain:
        for d in domain_terms:
            add(d)
        add(f"recent advances in {domain}")
        # Light topic cross for context
        for d in domain_terms[:2]:
            add(f"{d} {topic}")

    else:
        add(f"{topic} {domain}" if domain else topic)

    return queries


def generate_citation_map_queries(structured: Dict[str, Optional[str]], map_focus: str) -> List[str]:
    """Generate retrieval queries for citation map seed papers.

    Parameters
    ----------
    structured : dict
        Decomposed query with keys: topic, domain, aspect.
    map_focus : str
        One of: core_cluster, landscape, evolution.

    Returns
    -------
    list[str]
        List of query strings for seed paper search.
    """
    topic = structured.get("topic", "")
    domain = structured.get("domain")

    if map_focus == "core_cluster" and domain:
        queries = [
            f"{topic} {domain}",
            f"{domain} applications of {topic}",
        ]
    elif map_focus == "landscape" and domain:
        queries = [
            f"{topic} {domain}",
            f"{topic}",
            f"{domain}",
        ]
    elif map_focus == "evolution" and domain:
        queries = [
            f"{topic} {domain} survey",
            f"history of {topic} in {domain}",
            f"{topic} {domain}",
        ]
    else:
        queries = [f"{topic} {domain}" if domain else topic]

    return queries
