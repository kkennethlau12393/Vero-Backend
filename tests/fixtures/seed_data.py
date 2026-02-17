"""
Factory functions for realistic test data matching Alexandria schemas.
"""
from __future__ import annotations

from app.feature2.work_topic_store import WorkForMap, TopicScore, WorkPreview


def make_work(
    work_id: str = "W1234567890",
    title: str = "Attention Is All You Need",
    year: int = 2017,
    cited_by_count: int = 90000,
    authors_json: list | None = None,
    venue: str | None = "NeurIPS",
    primary_topic_id: str | None = "T100",
    primary_topic_score: float | None = 0.95,
    topics: list[TopicScore] | None = None,
    is_retracted: bool = False,
    abstract: str | None = "We propose a new simple network architecture...",
    category: str | None = "foundational",
    category_confidence: float | None = 0.9,
    referenced_works: list[str] | None = None,
    **overrides,
) -> WorkForMap:
    if authors_json is None:
        authors_json = ["Vaswani A.", "Shazeer N.", "Parmar N."]
    if topics is None:
        topics = [TopicScore(topic_id="T100", score=0.95)]
    return WorkForMap(
        work_id=work_id,
        title=title,
        year=year,
        cited_by_count=cited_by_count,
        authors_json=authors_json,
        venue=venue,
        primary_topic_id=primary_topic_id,
        primary_topic_score=primary_topic_score,
        topics=topics,
        is_retracted=is_retracted,
        abstract=abstract,
        category=category,
        category_confidence=category_confidence,
        referenced_works=referenced_works,
        **overrides,
    )


def make_work_preview(
    work_id: str = "W1234567890",
    title: str = "Attention Is All You Need",
    year: int = 2017,
    cited_by_count: int = 90000,
    authors: list[str] | None = None,
    venue: str | None = "NeurIPS",
) -> WorkPreview:
    if authors is None:
        authors = ["Vaswani A.", "Shazeer N."]
    return WorkPreview(
        work_id=work_id,
        title=title,
        year=year,
        cited_by_count=cited_by_count,
        authors=authors,
        venue=venue,
    )


def make_paper_batch(n: int = 10) -> list[WorkForMap]:
    """Generate n distinct papers with varied metadata."""
    papers = []
    for i in range(n):
        papers.append(make_work(
            work_id=f"W{2000000000 + i}",
            title=f"Paper {i}: Deep Learning in Domain {i}",
            year=2015 + (i % 10),
            cited_by_count=100 * (i + 1),
            primary_topic_id=f"T{200 + i % 5}",
            venue=["Nature", "Science", "NeurIPS", "ICML", "ACL"][i % 5],
            category=["foundational", "methodological", "applied", "recent", "handbook"][i % 5],
        ))
    return papers
