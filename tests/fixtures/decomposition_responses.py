"""
Factory functions for mock query decomposition responses.

Used by unit tests to create consistent, reproducible decomposition results
without calling the Groq LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def make_decomposition(
    topic: str = "natural language processing",
    topic_aliases: Optional[List[str]] = None,
    domain: Optional[str] = "law",
    domain_aliases: Optional[List[str]] = None,
    aspect: Optional[str] = None,
    aspect_aliases: Optional[List[str]] = None,
    suggested_specificity: str = "specific",
    reasoning: str = "Test decomposition",
) -> Dict[str, Any]:
    """Create a mock query decomposition result."""
    result: Dict[str, Any] = {
        "topic": topic,
        "topic_aliases": topic_aliases or [],
        "domain": domain,
        "domain_aliases": domain_aliases or [],
        "aspect": aspect,
        "aspect_aliases": aspect_aliases or [],
        "suggested_specificity": suggested_specificity,
        "reasoning": reasoning,
    }
    return result


def make_paper_decomposition(
    topic: str = "attention mechanisms",
    topic_aliases: Optional[List[str]] = None,
    domain: Optional[str] = "machine translation",
    domain_aliases: Optional[List[str]] = None,
    aspect: Optional[str] = "self-attention",
    aspect_aliases: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    suggested_specificity: str = "specific",
) -> Dict[str, Any]:
    """Create a mock paper decomposition result."""
    return {
        "topic": topic,
        "topic_aliases": topic_aliases or ["attention", "attention mechanism"],
        "domain": domain,
        "domain_aliases": domain_aliases or [],
        "aspect": aspect,
        "aspect_aliases": aspect_aliases or [],
        "keywords": keywords or ["attention", "transformer", "self-attention"],
        "suggested_specificity": suggested_specificity,
    }


def make_intersection_query() -> Dict[str, Any]:
    """NLP for law — canonical intersection query."""
    return make_decomposition(
        topic="natural language processing",
        topic_aliases=["NLP", "computational linguistics", "text mining"],
        domain="law",
        domain_aliases=["legal", "legal AI", "legaltech"],
        aspect=None,
        suggested_specificity="specific",
    )


def make_broad_query() -> Dict[str, Any]:
    """Transformer architectures — broad, no domain."""
    return make_decomposition(
        topic="transformer architectures",
        topic_aliases=["transformers", "attention models"],
        domain=None,
        aspect=None,
        suggested_specificity="broad",
    )


def make_specific_query() -> Dict[str, Any]:
    """BERT fine-tuning for medical NER — specific with aspect."""
    return make_decomposition(
        topic="BERT fine-tuning",
        topic_aliases=["BERT", "language model fine-tuning"],
        domain="medical",
        domain_aliases=["healthcare", "clinical", "biomedical"],
        aspect="named entity recognition",
        aspect_aliases=["NER", "entity extraction"],
        suggested_specificity="specific",
    )


def make_candidate_paper(
    title: str = "NLP Methods for Legal Text Analysis",
    abstract: str = "We apply natural language processing to legal text classification.",
    cited_by_count: int = 100,
    year: int = 2022,
    work_id: str = "W123",
) -> Dict[str, Any]:
    """Create a mock candidate paper for scoring tests."""
    return {
        "work_id": work_id,
        "title": title,
        "abstract": abstract,
        "cited_by_count": cited_by_count,
        "year": year,
    }
