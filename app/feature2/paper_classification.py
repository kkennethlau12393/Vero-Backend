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

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection

from .work_topic_store import WorkForMap

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5
MODEL_VERSION = "gpt-5-mini-2025-08-07"  # Fast model for regular classification
BORDERLINE_MODEL_VERSION = "gpt-5-mini-2025-08-07"  # Used for borderline fundamentals decisions
MAX_PARALLEL_BATCHES = 5

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


def _truncate_text(text: str, max_chars: int = 1000) -> str:
    """Truncate text to max characters, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def classify_borderline_llm(
    papers: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Call LLM to classify borderline papers for Fundamentals decision.

    This is used ONLY for papers that match textbook patterns but have
    subfield qualifiers. The LLM decides based on SEMANTIC INTENT:
    - Does this work derive general theory applicable across subfields?
    - Or does it assume existing theory and apply it to a specific domain?

    Returns dict mapping paper_id to {"category": str, "confidence": float}.
    """
    if not papers:
        return {}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found, skipping borderline LLM classification")
        return {}

    client = OpenAI(api_key=api_key)

    # Build papers JSON for prompt
    papers_json = json.dumps([
        {
            "id": p["paper_id"],
            "title": p["title"],
            "year": p.get("year"),
            "citations": p.get("citations", 0),
            "abstract": p.get("abstract", "")[:500],
        }
        for p in papers
    ], indent=2)

    prompt = f"""You are classifying academic papers to determine if they belong in the FUNDAMENTALS category.

FUNDAMENTALS criteria (VERY STRICT):
- Truly canonical, field-defining works that established core GENERAL theory
- Applicable across ALL subfields of the discipline
- Seminal textbooks taught in graduate courses across the field
- Examples: Chopra's "Dynamics of Structures", Timoshenko's vibration texts, Clough & Penzien

NOT FUNDAMENTALS (route elsewhere):
- Domain-specific textbooks that apply general theory to a particular subfield
- Books focused on specific applications (geotechnical, wind, offshore, etc.)
- Works that assume prior knowledge of fundamentals and extend to applications
- Even if highly cited, domain-specific works are NOT foundational

KEY QUESTION for each paper:
"Does this work DERIVE general theory/methods applicable across the entire field,
or does it APPLY existing theory to a specific domain?"

- If DERIVES general theory → "foundational"
- If APPLIES to specific domain → "applied" or "methodological"

For each paper, return:
- category: "foundational", "applied", "methodological", or "specific_topics"
- confidence: 0.0-1.0

Return ONLY a JSON object.
Example: {{"W123": {{"category": "applied", "confidence": 0.85}}}}

Papers to classify:
{papers_json}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=BORDERLINE_MODEL_VERSION,
                messages=[
                    {"role": "system", "content": "You are an expert academic classifier. Be STRICT about Fundamentals - only truly canonical, field-wide works qualify. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                timeout=60.0,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            result = json.loads(content)
            if isinstance(result, dict):
                valid_categories = {"foundational", "methodological", "applied", "specific_topics"}
                validated = {}
                for pid, data in result.items():
                    if isinstance(data, dict):
                        cat = data.get("category", "").lower()
                        conf = data.get("confidence", 0.7)
                        if cat in valid_categories:
                            validated[pid] = {
                                "category": cat,
                                "confidence": max(0.0, min(1.0, float(conf))),
                            }
                            logger.debug(f"Borderline LLM: {pid} -> {cat}")
                return validated

            return {}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse borderline LLM JSON: {e}")
            return {}
        except Exception as e:
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str or "connection" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"Borderline LLM classification failed: {e}")
            return {}

    return {}


def classify_batch_llm(
    papers: List[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """Call LLM to classify a batch of papers.

    Returns dict mapping paper_id to {"category": str, "confidence": float}.
    """
    if not papers:
        return {}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found, skipping LLM classification")
        return {}

    client = OpenAI(api_key=api_key)

    # Build papers JSON for prompt (include year AND citations for better classification)
    papers_json = json.dumps([
        {
            "id": p["paper_id"],
            "title": p["title"],
            "year": p.get("year"),
            "citations": p.get("cited_by_count", 0),
            "abstract": (p.get("abstract") or "")[:500]  # Truncate long abstracts
        }
        for p in papers
    ], indent=2)

    prompt = f"""Classify each academic paper into ONE category. USE THE YEAR FIELD - it's critical.

HARD RULES (follow these strictly):
- Papers with "review", "survey", "state-of-the-art", "meta-analysis" in title → HANDBOOK (never FOUNDATIONAL)
- Papers from 2018 or later → RECENT (unless clearly a review → HANDBOOK)
- Papers from 2016-2017 → Usually APPLIED or METHODOLOGICAL (too recent for FOUNDATIONAL)

CATEGORIES:

1. FOUNDATIONAL (EXTREMELY RARE - maybe 1-2 papers per batch):
   STRICT REQUIREMENTS - ALL must be true:
   - Published BEFORE 2010 (truly foundational work needs decades to prove impact)
   - 1000+ citations minimum
   - Universally taught across the ENTIRE field, not just a subfield
   - Original work that INVENTED methods, not reviews/surveys

   Examples: Newmark's time integration (1959), Chopra's Dynamics of Structures, Clough & Penzien

   NEVER FOUNDATIONAL:
   - Reviews, surveys, handbooks (→ HANDBOOK)
   - Domain-specific textbooks like "Geotechnical X" or "Wind Engineering" (→ APPLIED)
   - Papers after 2015 (→ RECENT or APPLIED)

2. METHODOLOGICAL: Method development papers (any era).
   - Papers that develop NEW techniques, algorithms, numerical methods
   - Improvements to existing methods
   - Must be about METHOD development, not application
   - Classic method papers from any year go here (not FOUNDATIONAL unless truly seminal)

3. HANDBOOK: Reviews, surveys, meta-analyses (any era).
   - Title contains: review, survey, overview, state-of-the-art, meta-analysis, handbook
   - Literature syntheses
   - NEVER put reviews in FOUNDATIONAL

4. RECENT: Papers from 2018 or later (STRICT YEAR CHECK).
   - ANY paper with year >= 2018 goes here (unless it's a review → HANDBOOK)
   - ML/AI/deep learning applications
   - Modern developments, current best practices
   - This is a YEAR-BASED category, not content-based

5. APPLIED: Domain-specific work from before 2018.
   - Empirical applications to specific domains
   - Domain-specific textbooks (geotechnical, wind, offshore, seismic)
   - Case studies, practical applications
   - Papers from 2010-2017 often go here

6. SPECIFIC_TOPICS: Narrow niche papers.
   - Very specialized, single-case studies
   - Conference proceedings
   - Too narrow for general interest

7. IMPLEMENTATION: Software packages, tools, tutorials.

DECISION TREE:
1. Is year >= 2018? → RECENT (unless review → HANDBOOK)
2. Is it a review/survey/handbook? → HANDBOOK
3. Is it pre-2010, 1000+ citations, universally foundational? → FOUNDATIONAL
4. Is it about method development? → METHODOLOGICAL
5. Is it domain-specific application? → APPLIED
6. Is it very narrow/niche? → SPECIFIC_TOPICS

Return JSON: {{"paper_id": {{"category": "...", "confidence": 0.X}}}}

Papers:
{papers_json}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": "You are an academic paper classifier. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                timeout=60.0,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            result = json.loads(content)
            if isinstance(result, dict):
                # Validate categories
                valid_categories = {"foundational", "methodological", "applied", "recent", "implementation", "handbook", "specific_topics"}
                validated = {}
                for pid, data in result.items():
                    if isinstance(data, dict):
                        cat = data.get("category", "").lower()
                        conf = data.get("confidence", 0.7)
                        if cat in valid_categories:
                            validated[pid] = {
                                "category": cat,
                                "confidence": max(0.0, min(1.0, float(conf))),
                            }
                return validated

            return {}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM classification JSON: {e}")
            return {}
        except Exception as e:
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str or "connection" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning(f"LLM classification failed: {e}")
            return {}

    return {}


def save_classifications_to_db(
    conn: Connection,
    classifications: Dict[str, Dict[str, Any]],
) -> None:
    """Save paper classifications to works.category column."""
    if not classifications:
        return

    try:
        saved_count = 0
        for paper_id, data in classifications.items():
            category = data.get("category", "applied")
            confidence = data.get("confidence", 0.7)
            result = conn.execute(
                text("""
                    UPDATE works
                    SET category = :category,
                        category_confidence = :confidence
                    WHERE work_id = :work_id
                """),
                {
                    "work_id": paper_id,
                    "category": category,
                    "confidence": confidence,
                },
            )
            if result.rowcount > 0:
                saved_count += 1
        # No explicit commit - let caller's transaction context handle it
        logger.info(f"Updated {saved_count}/{len(classifications)} paper categories")
    except Exception as e:
        logger.warning(f"Failed to save classifications to DB: {e}", exc_info=True)


def get_paper_categories(
    conn: Connection,
    paper_ids: List[str],
    works: Dict[str, WorkForMap],
    query_hash: str = None,  # Kept for API compatibility, not used
) -> Dict[str, PaperClassification]:
    """Main entry point for getting paper categories.

    Classification flow:
    1. Apply heuristics as FIRST-PASS FILTERS
       - Some results are final (implementation, handbook, method papers)
       - Some are borderline (textbooks with qualifiers) → need LLM review
    2. For borderline cases, call LLM with semantic intent prompt
    3. If no heuristic match, check cache
    4. Call regular LLM for remaining uncached papers

    Returns dict mapping paper_id to PaperClassification.
    Papers without a category will not be in the returned dict.
    """
    if not paper_ids:
        return {}

    result: Dict[str, PaperClassification] = {}
    needs_borderline_llm: List[str] = []  # Textbooks with qualifiers
    needs_regular_llm: List[str] = []  # No heuristic match
    heuristic_updates: Dict[str, Dict[str, Any]] = {}
    heuristic_final_count = 0
    heuristic_borderline_count = 0
    cached_count = 0

    # Step 1: Apply heuristics as first-pass filters
    for pid in paper_ids:
        work = works.get(pid)
        if not work:
            continue

        # Try heuristic classification
        heuristic_class, needs_llm_review = classify_paper_heuristic(work)

        if heuristic_class and not needs_llm_review:
            # Final heuristic decision (implementation, handbook, method paper, etc.)
            result[pid] = heuristic_class
            if work.category != heuristic_class.category.value:
                heuristic_updates[pid] = {
                    "category": heuristic_class.category.value,
                    "confidence": heuristic_class.confidence,
                }
            heuristic_final_count += 1

        elif heuristic_class and needs_llm_review:
            # Borderline case (textbook with qualifiers) - needs LLM semantic review
            # Store the heuristic suggestion but mark for LLM review
            needs_borderline_llm.append(pid)
            heuristic_borderline_count += 1

        elif work.category:
            # No heuristic match - check cache
            try:
                category = PaperCategory(work.category)
                result[pid] = PaperClassification(
                    paper_id=pid,
                    category=category,
                    confidence=work.category_confidence or 0.8,
                    source="cached",
                )
                cached_count += 1
            except ValueError:
                # Invalid category value, needs LLM reclassification
                needs_regular_llm.append(pid)
        else:
            # No heuristic, no cache - needs regular LLM
            needs_regular_llm.append(pid)

    if heuristic_final_count > 0:
        logger.info(f"Classified {heuristic_final_count} papers via heuristics (final)")
    if heuristic_borderline_count > 0:
        logger.info(f"Found {heuristic_borderline_count} borderline papers needing LLM review")
    if cached_count > 0:
        logger.info(f"Loaded {cached_count} cached classifications from works.category")

    # Save final heuristic updates to DB
    if heuristic_updates:
        logger.info(f"Updating {len(heuristic_updates)} cached classifications")
        save_classifications_to_db(conn, heuristic_updates)

    # Step 2: Call borderline LLM for papers with qualifiers (semantic intent check)
    if needs_borderline_llm:
        logger.info(f"Calling borderline LLM for {len(needs_borderline_llm)} textbooks with qualifiers")

        borderline_papers = []
        for pid in needs_borderline_llm:
            work = works.get(pid)
            if work:
                borderline_papers.append({
                    "paper_id": pid,
                    "title": work.title or "Untitled",
                    "year": work.year,
                    "citations": work.cited_by_count or 0,
                    "abstract": _truncate_text(work.abstract or ""),
                })

        borderline_results = classify_borderline_llm(borderline_papers)

        borderline_to_fundies = 0
        borderline_to_other = 0
        for pid, data in borderline_results.items():
            try:
                category = PaperCategory(data["category"])
                result[pid] = PaperClassification(
                    paper_id=pid,
                    category=category,
                    confidence=data["confidence"],
                    source="llm_borderline",
                )
                if category == PaperCategory.FOUNDATIONAL:
                    borderline_to_fundies += 1
                else:
                    borderline_to_other += 1
            except (ValueError, KeyError) as e:
                logger.debug(f"Invalid borderline LLM result for {pid}: {e}")
                # Fall back to regular LLM
                needs_regular_llm.append(pid)

        # Save borderline results
        if borderline_results:
            save_classifications_to_db(conn, borderline_results)

        logger.info(
            f"Borderline LLM: {borderline_to_fundies} → fundamentals, "
            f"{borderline_to_other} → other categories"
        )

        # Papers not in borderline_results need regular LLM
        for pid in needs_borderline_llm:
            if pid not in result:
                needs_regular_llm.append(pid)

    if not needs_regular_llm:
        logger.info(f"Total paper classifications: {len(result)}/{len(paper_ids)}")
        return result

    # Step 3: Call regular LLM for remaining papers
    logger.info(f"Calling regular LLM to classify {len(needs_regular_llm)} papers")

    # Prepare input for LLM (include year AND citations for better classification)
    papers_for_llm = []
    for pid in needs_regular_llm:
        work = works.get(pid)
        if not work:
            continue
        papers_for_llm.append({
            "paper_id": pid,
            "title": work.title or "Untitled",
            "year": work.year,
            "cited_by_count": work.cited_by_count or 0,
            "abstract": _truncate_text(work.abstract or ""),
        })

    # Split into batches and process in parallel
    batches = []
    for i in range(0, len(papers_for_llm), BATCH_SIZE):
        batches.append(papers_for_llm[i:i + BATCH_SIZE])

    llm_results: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        future_to_batch = {
            executor.submit(classify_batch_llm, batch): batch
            for batch in batches
        }

        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                llm_results.update(batch_results)
            except Exception as e:
                logger.error(f"Batch classification failed: {e}")

    # Process LLM results with post-processing rules
    llm_count = 0
    recent_overrides = 0
    for pid, data in llm_results.items():
        try:
            category = PaperCategory(data["category"])
            work = works.get(pid)
            paper_title = work.title if work else ""
            paper_year = work.year if work else None

            # Enforce year constraint for RECENT category
            # Papers must be 2018 or later to be classified as recent
            if category == PaperCategory.RECENT:
                if paper_year is None or paper_year < 2018:
                    # Override to APPLIED if year constraint not met
                    logger.debug(f"Override RECENT -> APPLIED for {pid} (year={paper_year})")
                    category = PaperCategory.APPLIED
                    data["category"] = "applied"
                    data["confidence"] = min(data["confidence"], 0.7)
                    recent_overrides += 1

            # Note: We no longer automatically promote textbooks to FOUNDATIONAL
            # The LLM prompt now correctly handles domain-specific textbooks

            result[pid] = PaperClassification(
                paper_id=pid,
                category=category,
                confidence=data["confidence"],
                source="llm",
            )
            llm_count += 1
        except (ValueError, KeyError) as e:
            logger.debug(f"Invalid LLM classification for {pid}: {e}")

    if recent_overrides > 0:
        logger.info(f"Overrode {recent_overrides} RECENT classifications due to year constraint")

    if llm_count > 0:
        logger.info(f"Classified {llm_count} papers via LLM")
        save_classifications_to_db(conn, llm_results)

    logger.info(f"Total paper classifications: {len(result)}/{len(paper_ids)}")
    return result
