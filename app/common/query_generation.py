"""
Multi-query generation for structured query retrieval.

Generates multiple targeted retrieval queries based on the decomposed
query structure and user's scope/focus preferences.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def generate_retrieval_queries(structured: Dict[str, Optional[str]], scope: str) -> List[str]:
    """Generate multiple retrieval queries based on decomposition and scope.

    Parameters
    ----------
    structured : dict
        Decomposed query with keys: topic, domain, aspect.
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

    if scope == "intersection" and domain:
        queries = [
            f"{topic} {domain}",
            f"{domain} applications of {topic}",
            f"{topic} applied to {domain}",
        ]
        if aspect:
            queries.append(f"{aspect} {topic} {domain}")

    elif scope == "broad" and domain:
        queries = [
            f"{topic} {domain}",
            f"{topic}",
            f"{domain} computational methods",
        ]

    elif scope == "topic_focused":
        queries = [
            f"{topic}",
            f"recent advances in {topic}",
        ]
        if domain:
            queries.append(f"{topic} {domain}")

    elif scope == "domain_focused" and domain:
        queries = [
            f"{domain} {topic}",
            f"AI methods for {domain}",
            f"computational approaches {domain}",
        ]

    else:
        queries = [f"{topic} {domain}" if domain else topic]

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
