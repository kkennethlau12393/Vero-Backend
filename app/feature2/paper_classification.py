"""
Paper classification module for ranking.

NOTE: This module contains legacy classification logic. The actual ranking output uses
5 categories defined in methodological_alignment.py:

ACTUAL OUTPUT CATEGORIES (used in production):
1. foundational - Seminal papers that defined the field
2. methodology - Method development papers
3. reviews - Survey papers, review articles, meta-analyses
4. applications - Domain-specific applications
5. textbooks - Educational/pedagogical resources

This module provides title pattern detection (HANDBOOK_PATTERNS, TEXTBOOK_PATTERNS)
used by methodological_alignment.py for category assignment.

The PaperCategory enum below is LEGACY and not used in production ranking output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .work_topic_store import WorkForMap

logger = logging.getLogger(__name__)

# Citation threshold for considering a paper potentially canonical
HIGH_CITATION_THRESHOLD = 500


# ============================================================================
# LEGACY: This enum is NOT used in production ranking output.
# Actual categories are defined in methodological_alignment.py:
#   foundational, methodology, reviews, applications, textbooks
# This enum is kept for backward compatibility with tests only.
# ============================================================================
class PaperCategory(Enum):
    """LEGACY: Categories of academic papers by methodological role.

    NOT USED IN PRODUCTION. See module docstring for actual output categories.
    """
    FOUNDATIONAL = "foundational"       # Canonical papers that established core methods
    METHODOLOGICAL = "methodological"   # Method development papers
    APPLIED = "applied"                 # Applications of methods to empirical questions
    RECENT = "recent"                   # UNUSED - removed from production output
    IMPLEMENTATION = "implementation"   # UNUSED - removed from production output
    HANDBOOK = "handbook"               # Mapped to "reviews" in production
    SPECIFIC_TOPICS = "specific_topics" # UNUSED - removed from production output


@dataclass
class PaperClassification:
    """Classification of a paper by methodological role."""
    paper_id: str
    category: PaperCategory
    confidence: float
    source: str  # "cached", "llm", "heuristic"


# Heuristic patterns for obvious cases (fast, no LLM needed)
IMPLEMENTATION_PATTERNS = [
    r'\bSTATA\s+module\b',
    r'\bSTATA\s+command\b',
    r'\bR\s+package\b',
    r'\bPython\s+library\b',
    r'\bPython\s+package\b',
    r'\bsoftware\s+package\b',
    r'\bPSMATCH\d*\b',
    r'\bXTIVREG\d*\b',
    r'\bXTDIDREGRESS\b',
    r'\breghdfe\b',
    r'\bivregress\b',
    r'\bfelm\b',
    r'\bplm\b',
    r'\blfe\b',
    r':\s*Stata\s+module\b',
    r':\s*R\s+package\b',
]

HANDBOOK_PATTERNS = [
    r'^Handbook\s+of\b',
    r'^The\s+Handbook\s+of\b',
    r'\bHandbook\s+on\b',
    r'^Review\s+of\b',
    r'^A\s+Review\s+of\b',
    r'^Survey\s+of\b',
    r'^A\s+Survey\s+of\b',
    r'\bMeta-analysis\b',
    r'\bMeta\s+analysis\b',
    r'\bSystematic\s+review\b',
    r'\bLiterature\s+review\b',
    # Additional patterns for reviews/surveys in middle of title
    r'\bstate[- ]of[- ]the[- ]art\s+review\b',
    r'\bstate[- ]of[- ]the[- ]art\s+survey\b',
    r'\bcritical\s+review\b',
    r'\bcomprehensive\s+review\b',
    r'\breview\s+and\s+analysis\b',
    r'\breview\s+of\s+(?:recent|current|existing)\b',
    r'\bsurvey\s+of\b',
    r'\boverview\s+of\b',
    r'\ba\s+review\b',  # "a review" anywhere in title
]

# Textbook patterns - these are FIRST-PASS FILTERS, not final decisions
# Papers matching these patterns are flagged for further evaluation
# Only truly canonical, field-wide textbooks should be FOUNDATIONAL
TEXTBOOK_PATTERNS = [
    r'^Fundamentals\s+of\b',
    r'^Introduction\s+to\b',
    r'^Principles\s+of\b',
    r'^Theory\s+of\b',
    r'^Dynamics\s+of\b',
    r'^Structural\s+Dynamics\b',
    r'^Vibration\s+of\b',
    r'^Mechanics\s+of\b',
    r'^Elements\s+of\b',
    r'^Foundations\s+of\b',
    r'^Engineering\s+Vibration',
    # Common textbook title patterns (can appear after colon or in subtitle)
    r':\s*Concepts\s+and\s+Applications',
    r':\s*Theory\s+and\s+Applications',
    r':\s*Principles\s+and\s+Practice',
    r':\s*A\s+Modern\s+Approach',
    r',\s*\d+(?:st|nd|rd|th)\s+Edition',  # "3rd Edition" etc.
    # "An Introduction to" anywhere in title (catches subtitles like "Speech Processing: An Introduction to...")
    r':\s*An?\s+Introduction\s+to\b',
    r'\bAn?\s+Introduction\s+to\b',
]

# Subfield qualifier patterns - these indicate domain-specific works
# Papers with textbook patterns + these qualifiers need LLM review
# They may be domain-specific textbooks, not general foundational works
SUBFIELD_QUALIFIER_PATTERNS = [
    # Geotechnical / foundation engineering
    r'\bgeotechnical\b',
    r'\bsoil\b',
    r'\bfoundation\s+engineering\b',
    r'\bpile\b',
    r'\bretaining\s+wall\b',
    # Wind / environmental
    r'\bwind\s+(loading|engineering|effects)\b',
    r'\boffshore\b',
    r'\bmarine\b',
    r'\bcoastal\b',
    r'\bhydraulic\b',
    r'\benvironmental\b',
    # Computational / numerical focus
    r'\bcomputational\b',
    r'\bnumerical\b',
    r'\bfinite\s+element\b',
    r'\bFEM\b',
    r'\bboundary\s+element\b',
    # Experimental / applied focus
    r'\bexperimental\b',
    r'\bapplied\b',
    # Materials-specific
    r'\bconcrete\b',
    r'\bsteel\b',
    r'\bmasonry\b',
    r'\btimber\b',
    r'\bcomposite\b',
    # Structure-specific
    r'\bbridge\b',
    r'\bhigh[- ]rise\b',
    r'\btower\b',
    r'\bdam\b',
    r'\btunnel\b',
]

# Method/algorithm paper patterns - these should be METHODOLOGICAL, not FOUNDATIONAL
# Original methods papers go to Core Concepts, not Fundamentals
METHOD_PAPER_PATTERNS = [
    r'\balgorithm\s+for\b',
    r'\bmethod\s+for\b',
    r'\bimproved\b.*\b(method|algorithm|scheme)\b',
    r'\bnumerical\b.*\b(method|scheme|algorithm|integration)\b',
    r'\btime\s+integration\b',
    r'\bcomponent[- ]mode\b',
    r'\bdissipation\b.*\balgorithm\b',
    r'\biterative\b.*\bmethod\b',
    r'\bconvergence\b.*\b(method|algorithm)\b',
    r'\bnew\b.*\b(method|technique|approach|formulation)\b',
    r'\befficient\b.*\b(method|algorithm|computation)\b',
]

# ML/AI patterns - papers with these + year >= 2018 should be RECENT
RECENT_ML_PATTERNS = [
    r'\bneural\s+network\b',
    r'\bdeep\s+learning\b',
    r'\bmachine\s+learning\b',
    r'\bPINN\b',
    r'\bphysics[- ]informed\b',
    r'\bCNN\b',
    r'\bLSTM\b',
    r'\bGAN\b',
    r'\breinforcement\s+learning\b',
    r'\btransformer\b',
    r'\battention\s+mechanism\b',
    r'\bconvolutional\b',
    r'\brecurrent\b',
    r'\bautoencoder\b',
]

# Conference proceedings patterns - should not be in main categories
PROCEEDINGS_PATTERNS = [
    r'\bproceedings\s+of\b',
    r'\bconference\s+proceedings\b',
    r'\b\d+(?:st|nd|rd|th)\s+(?:world\s+)?conference\b',
    r'\bsymposium\s+on\b',
    r'\bworkshop\s+on\b',
]


def is_textbook(title: str) -> bool:
    """Check if a title matches textbook patterns.

    Note: This is a FIRST-PASS filter. Textbooks with subfield qualifiers
    need LLM review to determine if they're truly foundational or domain-specific.
    """
    if not title:
        return False
    for pattern in TEXTBOOK_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


def has_subfield_qualifiers(title: str) -> bool:
    """Check if a title has domain-specific qualifiers.

    Papers with subfield qualifiers may be domain-specific textbooks
    (e.g., "Fundamentals of Geotechnical Engineering") rather than
    truly canonical field-wide works.
    """
    if not title:
        return False
    for pattern in SUBFIELD_QUALIFIER_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


def is_method_paper(title: str) -> bool:
    """Check if a title indicates a method/algorithm paper.

    Method papers should be METHODOLOGICAL, not FOUNDATIONAL,
    even if highly cited. They develop tools, not theory.
    """
    if not title:
        return False
    for pattern in METHOD_PAPER_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


def has_ml_patterns(title: str) -> bool:
    """Check if a title has ML/AI patterns.

    Papers with ML patterns + year >= 2018 should be RECENT.
    """
    if not title:
        return False
    for pattern in RECENT_ML_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


def is_proceedings(title: str) -> bool:
    """Check if a title is a conference proceedings.

    Proceedings should be excluded or routed to specific_topics.
    """
    if not title:
        return False
    for pattern in PROCEEDINGS_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False


@dataclass
class HeuristicResult:
    """Result of heuristic classification - may be final or need LLM review."""
    category: Optional[PaperCategory]
    confidence: float
    needs_llm_review: bool = False


def classify_paper_heuristic(work: WorkForMap) -> Tuple[Optional[PaperClassification], bool]:
    """Apply heuristic rules as first-pass filters.

    Returns:
        Tuple of (PaperClassification or None, needs_borderline_llm_review)
        - If classification is returned with needs_review=False: final decision
        - If classification is returned with needs_review=True: suggestion, needs LLM
        - If None is returned: no heuristic match, needs regular LLM
    """
    title = work.title or ""
    citations = work.cited_by_count or 0
    year = work.year

    # 1. Check implementation patterns (final decision)
    for pattern in IMPLEMENTATION_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return PaperClassification(
                paper_id=work.work_id,
                category=PaperCategory.IMPLEMENTATION,
                confidence=0.95,
                source="heuristic",
            ), False

    # 2. Check proceedings (route to specific_topics, final)
    if is_proceedings(title):
        return PaperClassification(
            paper_id=work.work_id,
            category=PaperCategory.SPECIFIC_TOPICS,
            confidence=0.85,
            source="heuristic",
        ), False

    # 3. Check handbook patterns (final decision, but not if it's a textbook)
    if not is_textbook(title):
        for pattern in HANDBOOK_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                return PaperClassification(
                    paper_id=work.work_id,
                    category=PaperCategory.HANDBOOK,
                    confidence=0.90,
                    source="heuristic",
                ), False

    # 4. Check method/algorithm papers (route to METHODOLOGICAL, final)
    if is_method_paper(title):
        return PaperClassification(
            paper_id=work.work_id,
            category=PaperCategory.METHODOLOGICAL,
            confidence=0.85,
            source="heuristic",
        ), False

    # 5. Check ML papers with recent year (route to RECENT, final)
    if has_ml_patterns(title) and year and year >= 2018:
        return PaperClassification(
            paper_id=work.work_id,
            category=PaperCategory.RECENT,
            confidence=0.85,
            source="heuristic",
        ), False

    # 6. Check textbook patterns - THIS IS WHERE THE KEY LOGIC IS
    if is_textbook(title):
        has_qualifiers = has_subfield_qualifiers(title)

        if has_qualifiers:
            # Textbook with subfield qualifiers (e.g., "Fundamentals of Geotechnical Engineering")
            # This is BORDERLINE - needs LLM to determine if truly foundational or domain-specific
            # Return a suggestion but flag for LLM review
            logger.debug(f"Borderline textbook with qualifiers: {title}")
            return PaperClassification(
                paper_id=work.work_id,
                category=PaperCategory.FOUNDATIONAL,  # Suggestion
                confidence=0.60,  # Low confidence - borderline
                source="heuristic_borderline",
            ), True  # Needs LLM review

        else:
            # Textbook WITHOUT qualifiers - more likely to be canonical
            # But still check citations for confidence
            if citations >= HIGH_CITATION_THRESHOLD:
                # High-citation textbook without qualifiers = likely canonical
                return PaperClassification(
                    paper_id=work.work_id,
                    category=PaperCategory.FOUNDATIONAL,
                    confidence=0.90,
                    source="heuristic",
                ), False
            else:
                # Lower citation textbook - might be recent edition or niche
                # Mark for LLM review to be safe
                return PaperClassification(
                    paper_id=work.work_id,
                    category=PaperCategory.FOUNDATIONAL,  # Suggestion
                    confidence=0.70,
                    source="heuristic_borderline",
                ), True  # Needs LLM review

    return None, False
