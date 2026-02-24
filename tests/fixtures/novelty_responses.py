"""
Factory functions and saved responses for Feature 3 (Novelty Assessment) tests.

Provides test data builders and saved API/LLM responses for deterministic testing.
"""
import json


# ============================================================================
# Factory Functions
# ============================================================================

def make_work_data(
    work_id="W1234567890",
    title="Test Paper Title",
    year=2020,
    cited_by_count=500,
    abstract="This paper introduces a novel approach to solving complex problems using advanced techniques.",
    primary_topic_id="T10123",
    category="foundational",
    category_confidence=0.9,
    doi=None,
    arxiv_id=None,
    authors=None,
    venue="Nature",
    is_open_access=True,
    oa_status="gold",
    oa_pdf_url=None,
):
    """Build a work metadata dict matching load_work_data() output."""
    return {
        "work_id": work_id,
        "title": title,
        "year": year,
        "cited_by_count": cited_by_count,
        "authors": authors or ["Author A", "Author B"],
        "venue": venue,
        "abstract": abstract,
        "primary_topic_id": primary_topic_id,
        "category": category,
        "category_confidence": category_confidence,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "is_open_access": is_open_access,
        "oa_status": oa_status,
        "oa_pdf_url": oa_pdf_url,
    }


def make_reference_paper(
    work_id="W9999999901",
    title="Referenced Paper",
    year=2015,
    cited_by_count=200,
    abstract="A foundational method for the field.",
    category="foundational",
    authors=None,
):
    """Build a reference paper dict."""
    return {
        "work_id": work_id,
        "title": title,
        "year": year,
        "cited_by_count": cited_by_count,
        "abstract": abstract,
        "category": category,
        "relationship": "cited_reference",
        "authors": authors or ["Author A", "Author B"],
    }


def make_landmark_paper(
    work_id="W8888888801",
    title="Landmark Paper",
    year=2010,
    cited_by_count=5000,
    abstract="A seminal contribution to the field.",
    field_name="Computer Science",
    authors=None,
):
    """Build a landmark paper dict."""
    return {
        "work_id": work_id,
        "title": title,
        "year": year,
        "cited_by_count": cited_by_count,
        "abstract": abstract,
        "field_name": field_name,
        "relationship": "field_landmark",
        "authors": authors or ["Author C", "Author D"],
    }


def make_grounding_paper(
    work_id="W7777777701",
    title="Grounding Paper",
    year=2012,
    cited_by_count=1000,
    relationship="cited_reference",
    relevance="Introduced key technique used in this paper.",
    authors=None,
):
    """Build a grounding paper dict."""
    return {
        "work_id": work_id,
        "title": title,
        "year": year,
        "cited_by_count": cited_by_count,
        "relationship": relationship,
        "relevance": relevance,
        "authors": authors or ["Author E", "Author F"],
    }


def make_references(n=5):
    """Generate n reference papers with varied metadata."""
    refs = []
    for i in range(n):
        refs.append(make_reference_paper(
            work_id=f"W990000000{i}",
            title=f"Reference Paper {i+1}",
            year=2015 - i,
            cited_by_count=200 * (i + 1),
        ))
    return refs


def make_landmarks(n=5):
    """Generate n landmark papers with varied metadata."""
    landmarks = []
    for i in range(n):
        landmarks.append(make_landmark_paper(
            work_id=f"W880000000{i}",
            title=f"Landmark Paper {i+1}",
            year=2010 - i * 3,
            cited_by_count=5000 * (i + 1),
        ))
    return landmarks


# ============================================================================
# Saved LLM Responses
# ============================================================================

NOVELTY_LLM_RESPONSE_HIGH = json.dumps({
    "summary": "This paper introduces a novel self-attention mechanism that replaces recurrent layers entirely, enabling parallelized sequence modeling with superior performance on translation benchmarks.",
    "keywords": ["self-attention", "transformer", "sequence modeling", "parallelization", "machine translation"],
    "novelty_assessment": {
        "whats_new": "Introduces the self-attention mechanism as the sole computation layer, eliminating recurrence entirely.",
        "compared_to_prior_work": "Prior sequence models like [W9900000000] relied on recurrent connections, limiting parallelization.",
        "novelty_level": "high",
        "confidence": "high",
        "novelty_explanation": "This paper demonstrates a major advance by replacing recurrent architectures [W9900000000] with pure attention mechanisms, achieving state-of-the-art results on WMT benchmarks while enabling full parallelization [W9900000001].",
        "grounding_papers": [
            {"work_id": "W9900000000", "title": "Reference Paper 1", "year": 2015, "cited_by_count": 200, "relationship": "cited_reference", "relevance": "Established recurrent neural translation approach that this work supersedes."},
            {"work_id": "W9900000001", "title": "Reference Paper 2", "year": 2014, "cited_by_count": 400, "relationship": "cited_reference", "relevance": "Introduced attention augmentation for recurrent sequence models."},
            {"work_id": "W8800000000", "title": "Landmark Paper 1", "year": 2010, "cited_by_count": 5000, "relationship": "field_landmark", "relevance": "Established foundational neural network training methods widely adopted in this field."},
            {"work_id": "W8800000001", "title": "Landmark Paper 2", "year": 2007, "cited_by_count": 10000, "relationship": "field_landmark", "relevance": "Introduced deep learning representations for NLP tasks."},
            {"work_id": "W8800000002", "title": "Landmark Paper 3", "year": 2004, "cited_by_count": 15000, "relationship": "field_landmark", "relevance": "Established benchmark evaluation methodology for sequence models."},
        ],
    },
})

NOVELTY_LLM_RESPONSE_MEDIUM_REVIEW = json.dumps({
    "summary": "This paper presents a comprehensive synthesis of machine learning applications in healthcare, covering diagnostic tools, treatment optimization, and patient outcome prediction.",
    "keywords": ["machine learning", "healthcare", "review", "diagnostics", "treatment optimization"],
    "novelty_assessment": {
        "whats_new": "Synthesizes and organizes current knowledge on ML in healthcare across diagnostic, treatment, and prediction domains.",
        "compared_to_prior_work": "Builds upon earlier reviews [W9900000000] and individual studies [W9900000001] to present a unified overview.",
        "novelty_level": "medium",
        "confidence": "high",
        "novelty_explanation": "Classified as medium because this is a review paper (Q0). It synthesizes existing work from [W9900000000] and field landmarks [W8800000000] without introducing new methods or empirical findings.",
        "grounding_papers": [
            {"work_id": "W9900000000", "title": "Reference Paper 1", "year": 2015, "cited_by_count": 200, "relationship": "cited_reference", "relevance": "Earlier review of ML methods in clinical settings."},
            {"work_id": "W9900000001", "title": "Reference Paper 2", "year": 2014, "cited_by_count": 400, "relationship": "cited_reference", "relevance": "Key study applying deep learning to medical imaging."},
            {"work_id": "W8800000000", "title": "Landmark Paper 1", "year": 2010, "cited_by_count": 5000, "relationship": "field_landmark", "relevance": "Established neural network architectures for image classification used in diagnostics."},
            {"work_id": "W8800000001", "title": "Landmark Paper 2", "year": 2007, "cited_by_count": 10000, "relationship": "field_landmark", "relevance": "Introduced deep learning frameworks enabling modern healthcare AI."},
            {"work_id": "W8800000002", "title": "Landmark Paper 3", "year": 2004, "cited_by_count": 15000, "relationship": "field_landmark", "relevance": "Foundational work on statistical learning theory applied in healthcare models."},
        ],
    },
})

NOVELTY_LLM_RESPONSE_PIONEERING = json.dumps({
    "summary": "This paper introduces the transformer architecture, a novel neural network design based entirely on self-attention mechanisms that fundamentally changed natural language processing and machine learning.",
    "keywords": ["transformer", "self-attention", "encoder-decoder", "neural architecture", "natural language processing"],
    "novelty_assessment": {
        "whats_new": None,
        "compared_to_prior_work": None,
        "novelty_level": "pioneering",
        "confidence": "high",
        "novelty_explanation": "This paper creates an entirely new paradigm for sequence processing. Prior to this work, all competitive models relied on recurrence [W9900000000] or convolutions [W9900000001]. The transformer architecture [W8800000000] introduced a fundamentally different approach that became the universal backbone for NLP and beyond.",
        "grounding_papers": [
            {"work_id": "W9900000000", "title": "Reference Paper 1", "year": 2015, "cited_by_count": 200, "relationship": "cited_reference", "relevance": "Established dominant recurrent approach that transformers replaced."},
            {"work_id": "W9900000001", "title": "Reference Paper 2", "year": 2014, "cited_by_count": 400, "relationship": "cited_reference", "relevance": "Convolutional sequence models that preceded the attention-only approach."},
            {"work_id": "W8800000000", "title": "Landmark Paper 1", "year": 2010, "cited_by_count": 5000, "relationship": "field_landmark", "relevance": "Established foundational deep learning training techniques."},
            {"work_id": "W8800000001", "title": "Landmark Paper 2", "year": 2007, "cited_by_count": 10000, "relationship": "field_landmark", "relevance": "Introduced word representations that enabled neural NLP."},
            {"work_id": "W8800000002", "title": "Landmark Paper 3", "year": 2004, "cited_by_count": 15000, "relationship": "field_landmark", "relevance": "Foundational neural network training methodology."},
        ],
    },
})

NOVELTY_LLM_RESPONSE_MALFORMED = '```json\n{"summary": "truncated response'


def make_groq_chat_response(content):
    """Build a mock object mimicking Groq/OpenAI chat completion response."""
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 500
    mock_response.usage.completion_tokens = 200
    return mock_response


# ============================================================================
# OpenAlex Mock Responses
# ============================================================================

OPENALEX_WORK_RESPONSE = {
    "id": "https://openalex.org/W1234567890",
    "title": "Test Paper Title",
    "publication_year": 2020,
    "cited_by_count": 500,
    "doi": "https://doi.org/10.1234/test.2020",
    "abstract_inverted_index": {
        "This": [0], "paper": [1], "introduces": [2], "a": [3],
        "novel": [4], "approach": [5], "to": [6], "solving": [7],
        "complex": [8], "problems": [9],
    },
    "primary_topic": {"id": "T10123", "display_name": "Machine Learning"},
    "authorships": [
        {"author": {"display_name": "Author A"}},
        {"author": {"display_name": "Author B"}},
    ],
}

OPENALEX_TOPIC_RESPONSE = {
    "id": "T10123",
    "display_name": "Machine Learning",
    "subfield": {"display_name": "Artificial Intelligence"},
    "field": {"display_name": "Computer Science"},
}


# ============================================================================
# Impact Analysis Mock Responses
# ============================================================================

IMPACT_LLM_RESPONSE_PARADIGM_SHIFT = json.dumps({
    "paper_type": "foundational",
    "before_approach": "Sequential processing with recurrent neural networks",
    "after_approach": "Parallel processing with self-attention mechanisms",
    "is_paradigm_shift": True,
    "shift_description": "Replaced recurrent processing with parallelized attention, transforming how the field approaches sequence modeling.",
})

IMPACT_LLM_RESPONSE_NO_SHIFT = json.dumps({
    "paper_type": "empirical",
    "before_approach": "Standard gradient descent optimization",
    "after_approach": "Standard gradient descent optimization with minor refinements",
    "is_paradigm_shift": False,
    "shift_description": None,
})

IMPACT_LLM_RESPONSE_SOFTWARE = json.dumps({
    "paper_type": "software",
    "before_approach": "Various individual ML implementations",
    "after_approach": "Unified ML framework for practitioners",
    "is_paradigm_shift": False,
    "shift_description": None,
})
