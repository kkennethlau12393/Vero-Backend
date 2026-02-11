"""
Paper-level impact analytics for Feature 3.

This module analyzes whether a specific paper represents a paradigm shift
in its field by comparing methodology in papers it cites vs papers that cite it.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from app.feature3.json_utils import extract_json_from_llm_response

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

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

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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
) -> float:
    """
    Calculate impact score based on citation velocity vs predecessors.

    Impact score = target_citations / avg(reference_citations)

    Higher scores indicate the paper had outsized impact relative to its
    sources, which can indicate a paradigm shift.
    """
    if not references:
        return 1.0  # No references to compare against

    ref_citations = [r.get("cited_by_count") or 0 for r in references]
    avg_ref_citations = sum(ref_citations) / len(ref_citations) if ref_citations else 1

    if avg_ref_citations < 1:
        avg_ref_citations = 1  # Avoid division by zero

    impact_score = target_cited_by_count / avg_ref_citations

    # Normalize to 0-1 range (sigmoid-like scaling)
    # Score of 3.0 maps to ~0.75, score of 10.0 maps to ~0.95
    normalized = impact_score / (impact_score + 3.0)

    return round(normalized, 3)


def _build_impact_prompt(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    references: List[Dict[str, Any]],
    citing_papers: List[Dict[str, Any]],
) -> str:
    """Build the LLM prompt for impact analysis."""
    year_str = f" ({year})" if year else ""
    abstract_text = _truncate_text(abstract or "No abstract available.", 500)

    # Build references section (papers it cites)
    refs_lines = []
    for i, ref in enumerate(references[:8], 1):
        ref_title = ref.get("title") or "Untitled"
        ref_year = ref.get("year") or "?"
        ref_abstract = _truncate_text(ref.get("abstract") or "", 150)
        refs_lines.append(f"{i}. {ref_title} ({ref_year})")
        if ref_abstract:
            refs_lines.append(f"   {ref_abstract}")
    refs_section = "\n".join(refs_lines) if refs_lines else "(No references available)"

    # Build citing papers section (papers that cite it)
    citing_lines = []
    for i, citer in enumerate(citing_papers[:8], 1):
        citer_title = citer.get("title") or "Untitled"
        citer_year = citer.get("year") or "?"
        citer_abstract = _truncate_text(citer.get("abstract") or "", 150)
        citing_lines.append(f"{i}. {citer_title} ({citer_year})")
        if citer_abstract:
            citing_lines.append(f"   {citer_abstract}")
    citing_section = "\n".join(citing_lines) if citing_lines else "(No citing papers available)"

    prompt = f"""Analyze this paper's impact on its field.

## TARGET PAPER
Title: {title}{year_str}
Abstract: {abstract_text}

## PAPERS IT CITES (before)
{refs_section}

## PAPERS THAT CITE IT (after)
{citing_section}

---

First, classify the paper type. Then analyze if it caused a paradigm shift.

Return JSON:
{{
    "paper_type": "software" | "review" | "foundational" | "empirical" | "measurement",
    "before_approach": "Dominant methodology in papers it cites (1 sentence)",
    "after_approach": "Dominant methodology in papers that cite it (1 sentence)",
    "is_paradigm_shift": true | false,
    "shift_description": "If paradigm shift, describe what changed (1 sentence). null if not a paradigm shift."
}}

Paper type definitions:
- "software": Primarily describes a software tool, library, package, or web service (e.g., TensorFlow, BLAST, NAMD)
- "review": Review article, meta-analysis, guideline, consensus statement, or synthesis of existing work
- "foundational": Introduces new theory, method, algorithm, or conceptual framework
- "empirical": Reports original experimental results, clinical trials, or observational studies
- "measurement": Introduces or validates a measurement scale, questionnaire, or assessment instrument

Paradigm shift rules:
- ONLY "foundational" papers can be paradigm shifts
- Software papers implement existing methods - they are NOT paradigm shifts
- Review papers synthesize existing knowledge - they are NOT paradigm shifts
- A paradigm shift means the field fundamentally changed HOW it approaches problems
- NOT just "highly cited" or "important" - must change methodology

Answer ONLY with the JSON object, no additional text."""

    return prompt


def analyze_paper_impact(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    cited_by_count: int,
    references: List[Dict[str, Any]],
    citing_papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze if a paper represents a paradigm shift.

    Args:
        title: Paper title
        abstract: Paper abstract
        year: Publication year
        cited_by_count: Number of citations
        references: Papers this work cites
        citing_papers: Papers that cite this work

    Returns:
        PaperImpactAnalysis dict with:
        - is_paradigm_shift: bool
        - impact_score: float (0-1)
        - before_approach: Optional[str]
        - after_approach: Optional[str]
        - shift_description: Optional[str]
    """
    # Calculate quantitative impact score
    impact_score = calculate_impact_score(cited_by_count, references)

    # Skip LLM analysis if we don't have citing papers
    if not citing_papers:
        return {
            "is_paradigm_shift": False,
            "impact_score": impact_score,
            "paper_type": None,
            "before_approach": None,
            "after_approach": None,
            "shift_description": None,
        }

    # Use LLM for paper type classification AND methodology comparison
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, returning quantitative-only analysis")
        return {
            "is_paradigm_shift": False,
            "impact_score": impact_score,
            "paper_type": None,
            "before_approach": None,
            "after_approach": None,
            "shift_description": None,
        }

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    prompt = _build_impact_prompt(title, abstract, year, references, citing_papers)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=60.0,
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()

            result, error = extract_json_from_llm_response(content, expected_type="object")

            if result is None:
                logger.warning(f"Failed to parse impact analysis JSON: {error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                    continue
                break

            # Get LLM's paper type classification
            paper_type = result.get("paper_type", "foundational")
            llm_paradigm_shift = bool(result.get("is_paradigm_shift", False))

            # Only foundational papers can be paradigm shifts
            # Software, review, measurement, and empirical papers cannot
            is_constrained = paper_type in ("software", "review", "measurement")
            final_paradigm_shift = llm_paradigm_shift and not is_constrained

            if is_constrained and llm_paradigm_shift:
                logger.info(f"LLM classified as {paper_type} - overriding paradigm_shift to false")

            logger.info(f"Impact analysis complete: type={paper_type}, paradigm_shift={final_paradigm_shift}, score={impact_score}")

            return {
                "is_paradigm_shift": final_paradigm_shift,
                "impact_score": impact_score,
                "paper_type": paper_type,
                "before_approach": result.get("before_approach"),
                "after_approach": result.get("after_approach"),
                "shift_description": result.get("shift_description") if final_paradigm_shift else None,
            }

        except Exception as e:
            logger.warning(f"Impact analysis LLM call exception: {e}")
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str or "connection" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                continue
            break

    # Fallback: quantitative-only analysis (no paper type classification available)
    return {
        "is_paradigm_shift": False,
        "impact_score": impact_score,
        "paper_type": None,
        "before_approach": None,
        "after_approach": None,
        "shift_description": None,
    }
