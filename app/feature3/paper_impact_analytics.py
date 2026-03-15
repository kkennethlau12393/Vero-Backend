"""
Paper-level impact analytics for Feature 3.

Provides paper type detection (software/review) and quantitative impact scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Patterns for detecting software/tool papers (general, not paper-specific)
SOFTWARE_TITLE_PATTERNS = [
    r'\b(library|package|toolkit|toolbox|software|framework)\b',
    r'\b(in python|in r|for python|for r)\b',
    r':\s*(a|an)\s+\w+\s+(library|package|tool|toolkit)\b',
    r'\b(machine learning|deep learning)\s+in\s+\w+$',  # "X in Python" pattern
    r'\bweb\s+server[s]?\b',  # Web server papers are software
]

SOFTWARE_ABSTRACT_PATTERNS = [
    r'\bopen[\-\s]?source\s+(library|package|software|tool)\b',
    r'\bpython\s+(library|package|module)\b',
    r'\bwe\s+(introduce|present|describe)\s+(a|an|the)\s+\w*\s*(library|package|toolkit|software)\b',
    r'\bpip\s+install\b',
    r'\bavailable\s+(at|on)\s+(github|pypi|cran)\b',
    # Software availability patterns - require software/tool/server context
    r'\b(freely|publicly)\s+(available|accessible)\s+\w*\s*(software|tool|server|package|code)\b',
    r'\b(a|an|new)\s+(program|tool|package)\s+(called|named|for)\b',  # "a program called X", "new tool for"
    r'\bsource\s+code\s+(is\s+)?(available|provided)\b',  # "source code available"
    r'\bcan\s+be\s+(downloaded|obtained)\b',  # "can be downloaded from"
    # Web server patterns
    r'\bweb\s+server[s]?\b',  # Web servers are software
    r'\b(freely|publicly)\s+accessible\s+web\b',  # "freely accessible Web servers"
]

# Patterns for detecting review/guideline papers (general, not paper-specific)
REVIEW_TITLE_PATTERNS = [
    r'\b(review|survey|meta[\-\s]?analysis|overview|systematic review)\b',
    r'\b(guideline[s]?|statement|checklist|recommendation[s]?)\b',
    r'\b(preferred reporting|reporting items|PRISMA|CONSORT|STROBE)\b',
    r'\b(state[\-\s]of[\-\s]the[\-\s]art|advances in|progress in)\b',
    # Synthesis/perspective patterns - papers that define/summarize a field
    r'\bhallmarks\s+of\b',  # "Hallmarks of X" papers are definitional/synthesis works
    r'\b(revisited|the\s+next\s+generation)\b',  # Update papers referencing prior work
    r'\b(an?\s+)?update(d)?\b',  # "An update on X", "Updated guidelines"
    r'\bperspective(s)?\s+on\b',  # Perspective papers
    r'\bemerging\s+(concepts?|themes?|paradigms?)\b',  # Synthesis of emerging work
    r'^introduction\s+to\b',  # "Introduction to X" - editorial/overview papers
]

REVIEW_ABSTRACT_PATTERNS = [
    r'\b(systematic(ally)?\s+review|meta[\-\s]?analysis)\b',
    r'\b(guideline[s]?|statement|checklist|recommendation[s]?)\b',
    r'\b(we\s+systematically\s+(searched|reviewed))\b',
    r'\b(reporting\s+(guideline|standard|quality))\b',
    r'\b(preferred\s+reporting\s+items)\b',
    r'\b(comprehensive\s+(review|overview|survey))\b',
    # Synthesis language in abstracts - be specific to avoid methodology papers
    r'\b(we\s+(summarize|review)\s+(the\s+)?(literature|evidence|studies|findings|advances))\b',
    r'\b(this\s+review\s+(summarizes|reviews|discusses|outlines|covers))\b',  # "This review..." not "This paper reviews..."
    r'\b(emerging\s+(hallmarks?|concepts?|evidence))\b',
    r'\b(here\s+we\s+(review|summarize)\s+(the\s+)?(literature|evidence|studies))\b',
]


def _is_software_tool_paper(title: Optional[str], abstract: Optional[str]) -> bool:
    """
    Detect if a paper is primarily about a software tool/library.

    Software tools are NOT paradigm shifts - they implement existing methods.
    This uses general pattern matching, not paper-specific hardcoding.

    Returns True if the paper appears to be a software/tool paper.
    """
    import re

    text_to_check = f"{title or ''} {abstract or ''}".lower()

    # Check title patterns (more weight)
    if title:
        title_lower = title.lower()
        for pattern in SOFTWARE_TITLE_PATTERNS:
            if re.search(pattern, title_lower, re.IGNORECASE):
                logger.debug(f"Software paper detected via title pattern: {pattern}")
                return True

    # Check abstract patterns
    if abstract:
        abstract_lower = abstract.lower()
        for pattern in SOFTWARE_ABSTRACT_PATTERNS:
            if re.search(pattern, abstract_lower, re.IGNORECASE):
                logger.debug(f"Software paper detected via abstract pattern: {pattern}")
                return True

    return False


def _is_review_guideline_paper(title: Optional[str], abstract: Optional[str]) -> bool:
    """
    Detect if a paper is a review, meta-analysis, or guideline.

    Reviews/guidelines are NOT paradigm shifts - they synthesize existing knowledge.
    This uses general pattern matching, not paper-specific hardcoding.

    Returns True if the paper appears to be a review/guideline paper.
    """
    import re

    # Check title patterns (more weight)
    if title:
        title_lower = title.lower()
        for pattern in REVIEW_TITLE_PATTERNS:
            if re.search(pattern, title_lower, re.IGNORECASE):
                logger.debug(f"Review paper detected via title pattern: {pattern}")
                return True

    # Check abstract patterns
    if abstract:
        abstract_lower = abstract.lower()
        for pattern in REVIEW_ABSTRACT_PATTERNS:
            if re.search(pattern, abstract_lower, re.IGNORECASE):
                logger.debug(f"Review paper detected via abstract pattern: {pattern}")
                return True

    return False


def _truncate_text(text_val: str, max_chars: int = 300) -> str:
    """Truncate text to max characters, preserving word boundaries."""
    if not text_val:
        return ""
    if len(text_val) <= max_chars:
        return text_val
    truncated = text_val[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def calculate_impact_score(
    target_cited_by_count: int,
    references: List[Dict[str, Any]],
    is_paradigm_shift: bool = False,
) -> float:
    """
    Calculate impact score using three signals:

    1. Absolute impact (45%): log-scaled citation count, ceiling 20K.
    2. Breadth (10%): log-scaled number of references.
    3. LLM paradigm shift (45%): 1.0 if paradigm shift, 0.3 otherwise.

    No relative component: references passed here are curated timeline
    landmarks (high-citation seminal works), not the paper's actual
    bibliography. Comparing against them always tanks the ratio.

    When no references available, uses absolute + LLM only (55/45).

    Returns 0.0–1.0.
    """
    import math

    # -- Absolute impact (log-scaled, 20K ceiling) --
    abs_score = min(math.log10(max(target_cited_by_count, 1)) / math.log10(20_000), 1.0)

    # -- LLM assessment signal --
    llm_score = 1.0 if is_paradigm_shift else 0.3

    if not references:
        # No references available — absolute + LLM only
        return round(0.55 * abs_score + 0.45 * llm_score, 3)

    # -- Breadth (number of references, log-scaled) --
    num_refs = len(references) if references else 1
    breadth = min(math.log10(max(num_refs, 1)) / math.log10(100), 1.0)

    # -- Weighted combination --
    score = 0.45 * abs_score + 0.10 * breadth + 0.45 * llm_score

    return round(score, 3)
