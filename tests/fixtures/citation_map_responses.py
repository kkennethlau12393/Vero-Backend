"""
Saved API responses and factory functions for Feature 1 (Citation Map) tests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from app.feature1.schemas import CitationNode, CitationEdge, CitationMapRequest


# ============================================================================
# OpenAlex Responses
# ============================================================================

def make_openalex_work(
    work_id: str = "W100",
    title: str = "Test Paper",
    year: int = 2020,
    cited_by_count: int = 500,
    doi: str | None = None,
    abstract_inverted_index: dict | None = None,
    referenced_works: list[str] | None = None,
) -> dict:
    """Build a single OpenAlex work object."""
    result = {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "publication_year": year,
        "cited_by_count": cited_by_count,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "referenced_works": referenced_works or [],
    }
    if abstract_inverted_index is not None:
        result["abstract_inverted_index"] = abstract_inverted_index
    return result


def make_openalex_search_response(works: list[dict] | None = None) -> dict:
    """Wrap works in an OpenAlex search response envelope."""
    if works is None:
        works = [
            make_openalex_work("W100", "Attention Is All You Need", 2017, 90000),
            make_openalex_work("W101", "BERT: Pre-training", 2019, 60000),
            make_openalex_work("W102", "Deep Residual Learning", 2016, 150000),
        ]
    return {
        "meta": {"count": len(works), "db_response_time_ms": 30, "page": 1, "per_page": 200},
        "results": works,
    }


SEED_WORK_RESPONSE = make_openalex_work(
    work_id="W999",
    title="Attention Is All You Need",
    year=2017,
    cited_by_count=90000,
    doi="10.5555/3295222.3295349",
    abstract_inverted_index={"We": [0], "propose": [1], "a": [2], "new": [3], "architecture": [4]},
    referenced_works=[
        "https://openalex.org/W200",
        "https://openalex.org/W201",
        "https://openalex.org/W202",
    ],
)

CITING_PAPERS_RESPONSE = make_openalex_search_response([
    make_openalex_work("W300", "BERT: Pre-training of Deep Bidirectional Transformers", 2019, 60000),
    make_openalex_work("W301", "GPT-2: Language Models are Unsupervised Multitask Learners", 2019, 10000),
    make_openalex_work("W302", "Vision Transformer", 2021, 20000),
    make_openalex_work("W303", "Reformer: The Efficient Transformer", 2020, 1500),
    make_openalex_work("W304", "Longformer: The Long-Document Transformer", 2020, 3000),
])

REFERENCES_WORK_RESPONSE = {
    **SEED_WORK_RESPONSE,
    "referenced_works": [
        "https://openalex.org/W200",
        "https://openalex.org/W201",
        "https://openalex.org/W202",
    ],
}

REFERENCES_DETAILS_RESPONSE = make_openalex_search_response([
    make_openalex_work("W200", "Sequence to Sequence Learning", 2014, 20000),
    make_openalex_work("W201", "Neural Machine Translation by Jointly Learning to Align", 2015, 30000),
    make_openalex_work("W202", "Convolutional Sequence to Sequence Learning", 2017, 3000),
])


# ============================================================================
# Semantic Scholar Responses
# ============================================================================

def make_s2_paper(
    paper_id: str = "abc123",
    title: str = "Test Paper",
    year: int = 2020,
    citation_count: int = 500,
    doi: str | None = None,
    openalex_id: str | None = None,
) -> dict:
    """Build a single S2 paper object."""
    external_ids = {}
    if doi:
        external_ids["DOI"] = doi
    if openalex_id:
        external_ids["OpenAlex"] = openalex_id
    return {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "citationCount": citation_count,
        "externalIds": external_ids,
    }


S2_SEARCH_RESPONSE = {
    "total": 3,
    "data": [
        make_s2_paper("s2_1", "Attention Is All You Need", 2017, 90000, doi="10.5555/3295222.3295349", openalex_id="W999"),
        make_s2_paper("s2_2", "BERT", 2019, 60000, doi="10.1234/bert", openalex_id="W300"),
        make_s2_paper("s2_3", "GPT-2", 2019, 10000, doi="10.1234/gpt2"),
    ],
}

S2_CITATIONS_RESPONSE = {
    "data": [
        {"citingPaper": make_s2_paper("s2_c1", "Paper citing seed 1", 2020, 500, doi="10.1234/cite1")},
        {"citingPaper": make_s2_paper("s2_c2", "Paper citing seed 2", 2021, 200, doi="10.1234/cite2")},
        {"citingPaper": make_s2_paper("s2_c3", None, None, 0)},  # empty citing paper
    ],
}

S2_REFERENCES_RESPONSE = {
    "data": [
        {"citedPaper": make_s2_paper("s2_r1", "Referenced paper 1", 2015, 8000, doi="10.1234/ref1")},
        {"citedPaper": make_s2_paper("s2_r2", "Referenced paper 2", 2010, 15000, doi="10.1234/ref2")},
        {"citedPaper": None},  # null citedPaper
    ],
}

S2_EMPTY_RESPONSE = {"data": []}


# ============================================================================
# ArXiv Response
# ============================================================================

ARXIV_SEARCH_RESPONSE = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <published>2017-06-12T00:00:00Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1810.04805v2</id>
    <title>BERT: Pre-training of Deep Bidirectional Transformers</title>
    <published>2018-10-11T00:00:00Z</published>
  </entry>
</feed>'''

ARXIV_EMPTY_RESPONSE = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


# ============================================================================
# Groq LLM Responses for Citation Map
# ============================================================================

SEED_SCORING_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"P0": "ESSENTIAL", "P1": "HIGH", "P2": "MEDIUM", "P3": "LOW"}'
            }
        }
    ],
    "usage": {"prompt_tokens": 300, "completion_tokens": 50},
}

SEED_SCORING_ALL_NONE_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"P0": "NONE", "P1": "NONE", "P2": "NONE"}'
            }
        }
    ],
    "usage": {"prompt_tokens": 300, "completion_tokens": 50},
}

QUERY_EXPANSION_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '[{"title": "Attention Is All You Need", "year": 2017, "citations": 90000}]'
            }
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
}

GARBAGE_LLM_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "I cannot classify these papers. Here is a haiku instead."
            }
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
}


# ============================================================================
# Factory helpers
# ============================================================================

def make_paper_dict(
    work_id: str = "W100",
    title: str = "Test Paper",
    year: int = 2020,
    cited_by_count: int = 500,
    abstract: str | None = None,
    source: str = "openalex",
    hop: int = 0,
    is_seed: bool = False,
) -> dict:
    """Build a paper dict as used internally by citation_map_service."""
    return {
        "work_id": work_id,
        "title": title,
        "year": year,
        "cited_by_count": cited_by_count,
        "abstract": abstract,
        "source": source,
        "hop": hop,
        "is_seed": is_seed,
    }


def make_papers_sorted_by_cites(n: int = 30, base_cites: int = 10000) -> list[dict]:
    """Generate n papers sorted by citation count descending."""
    return [
        make_paper_dict(
            work_id=f"W{1000 + i}",
            title=f"Paper {i}",
            year=2015 + (i % 10),
            cited_by_count=base_cites - i * (base_cites // n),
        )
        for i in range(n)
    ]
