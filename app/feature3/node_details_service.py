"""
Node details service for Feature 3.

This module generates detailed pop-up information for a specific node
in the citation map, including LLM-generated summary, keywords, and
novelty assessment grounded in actual papers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.feature3.abstract_enrichment import ensure_valid_abstract
from app.feature3.grounding_supplement import supplement_grounding_papers, get_field_from_topic_id
from app.feature3.methodology import get_methodology, is_methodology_mismatch
from app.feature3.json_utils import extract_json_from_llm_response
from app.feature3.paper_identity import title_word_overlap, content_word_overlap
from app.feature3.landmark_retrieval import get_topic_landmarks
from app.feature3.node_timeline import build_node_timeline
from app.feature3.reference_store import get_referenced_works
from app.feature3.schemas import (
    ConnectedWork,
    GroundingPaper,
    NodeDetailsResponse,
    NodeTimeline,
    NoveltyAssessment,
    PaperImpactAnalysis,
    TimelinePaper,
    TimelineSection,
)
from app.feature3.topic_inference import ensure_topic
from app.feature3.topic_lookup import get_topic_display_name
from app.shared.pdf_utils import download_and_extract_pdf, extract_paper_sections

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Bump this when model OR prompt changes to auto-invalidate cached assessments
ASSESSMENT_VERSION = "maverick-v10"


def get_cached_details(conn: Connection, work_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached node details for a work."""
    try:
        row = conn.execute(
            text("""
                SELECT summary, keywords, novelty_assessment, model_version,
                       assessment_unavailable_reason
                FROM node_details_cache
                WHERE work_id = :work_id AND model_version = :model_version
            """),
            {"work_id": work_id, "model_version": ASSESSMENT_VERSION},
        ).mappings().first()

        if row:
            return {
                "summary": row["summary"],
                "keywords": row["keywords"] or [],
                "novelty_assessment": row["novelty_assessment"],  # Keep None if null
                "assessment_unavailable_reason": row.get("assessment_unavailable_reason"),
            }
        return None
    except Exception as e:
        logger.warning(f"Failed to retrieve cached details: {e}")
        return None


def cache_details(
    conn: Connection,
    work_id: str,
    summary: str,
    keywords: List[str],
    novelty_assessment: Optional[Dict[str, Any]],
    assessment_unavailable_reason: Optional[str] = None,
) -> None:
    """Store node details in cache."""
    try:
        conn.execute(
            text("""
                INSERT INTO node_details_cache (work_id, summary, keywords, novelty_assessment, model_version, assessment_unavailable_reason)
                VALUES (:work_id, :summary, :keywords, :novelty_assessment, :model_version, :assessment_unavailable_reason)
                ON CONFLICT (work_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    keywords = EXCLUDED.keywords,
                    novelty_assessment = EXCLUDED.novelty_assessment,
                    model_version = EXCLUDED.model_version,
                    assessment_unavailable_reason = EXCLUDED.assessment_unavailable_reason,
                    created_at = now()
            """),
            {
                "work_id": work_id,
                "summary": summary,
                "keywords": json.dumps(keywords),
                "novelty_assessment": json.dumps(novelty_assessment) if novelty_assessment else None,
                "model_version": ASSESSMENT_VERSION,
                "assessment_unavailable_reason": assessment_unavailable_reason,
            },
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to cache details: {e}")


def _truncate_text(text_val: str, max_chars: int = 500) -> str:
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


# ============================================================================
# Post-LLM Validation: Banned Verbs
# ============================================================================

import re as _re

# Banned verbs from the novelty prompt rubric (all forms)
_BANNED_VERB_PATTERNS = [
    # Order: match 3rd person ('s' suffix) BEFORE base/infinitive form
    # to avoid base form regex turning "evaluates" into "validatees"
    (r'\bexplores\b', 'demonstrates'),
    (r'\bexplore\b', 'demonstrate'),
    (r'\bexplored\b', 'demonstrated'),
    (r'\bexploring\b', 'demonstrating'),
    (r'\bdiscusses\b', 'presents'),
    (r'\bdiscuss\b', 'present'),
    (r'\bdiscussed\b', 'presented'),
    (r'\bdiscussing\b', 'presenting'),
    (r'\bexamines\b', 'establishes'),
    (r'\bexamine\b', 'establish'),
    (r'\bexamined\b', 'established'),
    (r'\bexamining\b', 'establishing'),
    (r'\binvestigates\b', 'demonstrates'),
    (r'\binvestigate\b', 'demonstrate'),
    (r'\binvestigated\b', 'demonstrated'),
    (r'\binvestigating\b', 'demonstrating'),
    (r'\bassesses\b', 'validates'),
    (r'\bassess\b', 'validate'),
    (r'\bassessed\b', 'validated'),
    (r'\bassessing\b', 'validating'),
    (r'\bevaluates\b', 'validates'),
    (r'\bevaluate\b', 'validate'),
    (r'\bevaluated\b', 'validated'),
    (r'\bevaluating\b', 'validating'),
    (r'\baddresses\b', 'presents'),
    (r'\baddress\b', 'present'),
    (r'\baddressed\b', 'presented'),
    (r'\baddressing\b', 'presenting'),
    (r'\blooks at\b', 'presents'),
    (r'\blook at\b', 'present'),
    (r'\blooked at\b', 'presented'),
    (r'\blooking at\b', 'presenting'),
    (r'\bstudies\b', 'demonstrates'),
    (r'\bstudy\b', 'demonstrate'),
    (r'\bstudied\b', 'demonstrated'),
    (r'\bstudying\b', 'demonstrating'),
    (r'\banalyzes\b', 'establishes'),
    (r'\banalyze\b', 'establish'),
    (r'\banalyzed\b', 'established'),
    (r'\banalyzing\b', 'establishing'),
    (r'\banalyses\b', 'establishes'),
    (r'\banalyse\b', 'establish'),
    (r'\banalysed\b', 'established'),
    (r'\banalysing\b', 'establishing'),
    # "reviews" as a verb (not "this review" as a noun)
    (r'\breviews\b(?!\s+(?:of|paper|article))', 'synthesizes'),
    (r'\breview\b(?=\s+(?:the|this|how|what|key|recent|current))', 'synthesize'),
    (r'\breviewed\b', 'synthesized'),
    (r'\breviewing\b', 'synthesizing'),
]

# Pre-compile for performance
_BANNED_COMPILED = [(_re.compile(pat, _re.IGNORECASE), repl) for pat, repl in _BANNED_VERB_PATTERNS]


def _scrub_banned_verbs(text: str) -> str:
    """Replace banned verbs with approved alternatives. Returns cleaned text."""
    if not text:
        return text
    result = text
    for pattern, replacement in _BANNED_COMPILED:
        result = pattern.sub(replacement, result)
    return result


def _has_banned_verbs(text: str) -> bool:
    """Check if text contains any banned verbs."""
    if not text:
        return False
    for pattern, _ in _BANNED_COMPILED:
        if pattern.search(text):
            return True
    return False


# Boilerplate relevance patterns that should be replaced
_BOILERPLATE_RELEVANCE = {
    "Landmark paper in this field for context",
    "Cited reference for methodology comparison",
    "Cited in novelty assessment narrative",
}


def _generate_relevance(paper: Dict[str, Any], relationship: str) -> str:
    """Generate a meaningful relevance string from paper metadata instead of boilerplate."""
    title = paper.get("title", "")
    year = paper.get("year")
    cites = paper.get("cited_by_count", 0) or 0

    # Build a relevance string from the paper's title
    if relationship == "field_landmark":
        if cites > 10000:
            return f"Established foundational work ({title[:60]}{'...' if len(title) > 60 else ''}) widely adopted in this field."
        elif year and year < 2000:
            return f"Introduced early methods ({title[:60]}{'...' if len(title) > 60 else ''}) that shaped this research area."
        else:
            return f"Contributed key advances ({title[:60]}{'...' if len(title) > 60 else ''}) relevant to this paper's domain."
    else:
        return f"Cited by this paper for its contribution: {title[:80]}{'...' if len(title) > 80 else ''}"


def _generate_summary_from_abstract(abstract: Optional[str], title: Optional[str] = None) -> str:
    """
    Generate a basic summary from the abstract when LLM assessment is unavailable.

    This provides a fallback summary derived directly from the paper's abstract,
    without any novelty claims or comparisons to other work.
    """
    if not abstract:
        if title:
            return f"This paper discusses: {title}"
        return "Summary unavailable - no abstract provided."

    # Take the first 2-3 sentences as a basic summary
    sentences = abstract.replace("\n", " ").split(". ")
    summary_sentences = []
    char_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Stop after ~400 chars or 3 sentences
        if char_count + len(sentence) > 400 or len(summary_sentences) >= 3:
            break
        summary_sentences.append(sentence)
        char_count += len(sentence)

    if not summary_sentences:
        return _truncate_text(abstract, 400)

    result = ". ".join(summary_sentences)
    if not result.endswith("."):
        result += "."
    return result


def _extract_keywords_from_abstract(abstract: Optional[str]) -> List[str]:
    """
    Extract basic keywords from abstract when LLM assessment is unavailable.

    This uses simple heuristics to identify potential keywords:
    - Capitalized multi-word phrases (likely proper nouns/technical terms)
    - Common academic signal words
    """
    if not abstract:
        return []

    import re

    keywords = set()

    # Look for quoted terms
    quoted = re.findall(r'"([^"]+)"', abstract)
    for term in quoted[:3]:
        if len(term) < 50:
            keywords.add(term.lower())

    # Look for capitalized phrases (2-3 words) that might be technical terms
    # Exclude sentence starters by looking for mid-sentence capitals
    cap_phrases = re.findall(r'(?<=[a-z]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', abstract)
    for phrase in cap_phrases[:5]:
        if len(phrase) > 3 and len(phrase) < 40:
            keywords.add(phrase.lower())

    # Common academic/technical patterns
    patterns = [
        r'\b(neural network[s]?)\b',
        r'\b(machine learning)\b',
        r'\b(deep learning)\b',
        r'\b(natural language processing)\b',
        r'\b(computer vision)\b',
        r'\b(reinforcement learning)\b',
        r'\b(transformer[s]?)\b',
        r'\b(attention mechanism[s]?)\b',
        r'\b(convolutional)\b',
        r'\b(recurrent)\b',
        r'\b(optimization)\b',
        r'\b(classification)\b',
        r'\b(regression)\b',
        r'\b(clustering)\b',
        r'\b(embedding[s]?)\b',
    ]

    abstract_lower = abstract.lower()
    for pattern in patterns:
        matches = re.findall(pattern, abstract_lower)
        for match in matches:
            keywords.add(match)

    return list(keywords)[:10]


def load_work_data(conn: Connection, work_id: str) -> Optional[Dict[str, Any]]:
    """Load work metadata from the works table."""
    row = conn.execute(
        text("""
            SELECT work_id, title, year, cited_by_count, authors_json, venue,
                   abstract, primary_topic_id, category, category_confidence,
                   doi, arxiv_id, is_open_access, oa_status, oa_pdf_url
            FROM works
            WHERE work_id = :work_id
        """),
        {"work_id": work_id},
    ).mappings().first()

    if not row:
        return None

    return {
        "work_id": row["work_id"],
        "title": row["title"],
        "year": row["year"],
        "cited_by_count": int(row["cited_by_count"] or 0),
        "authors": row["authors_json"] or [],
        "venue": row["venue"],
        "abstract": row["abstract"],
        "primary_topic_id": row["primary_topic_id"],
        "category": row["category"],
        "category_confidence": row["category_confidence"],
        "doi": row["doi"],
        "arxiv_id": row["arxiv_id"],
        "is_open_access": row["is_open_access"],
        "oa_status": row["oa_status"],
        "oa_pdf_url": row["oa_pdf_url"],
    }


def load_connected_works(
    conn: Connection, map_id: UUID, work_id: str
) -> List[ConnectedWork]:
    """Load papers connected to the target work via map edges."""
    # Papers this work cites (outgoing edges)
    cites_rows = conn.execute(
        text("""
            SELECT e.to_work_id, w.title
            FROM map_edges e
            LEFT JOIN works w ON w.work_id = e.to_work_id
            WHERE e.map_id = :map_id AND e.from_work_id = :work_id
        """),
        {"map_id": map_id, "work_id": work_id},
    ).mappings().all()

    # Papers that cite this work (incoming edges)
    cited_by_rows = conn.execute(
        text("""
            SELECT e.from_work_id, w.title
            FROM map_edges e
            LEFT JOIN works w ON w.work_id = e.from_work_id
            WHERE e.map_id = :map_id AND e.to_work_id = :work_id
        """),
        {"map_id": map_id, "work_id": work_id},
    ).mappings().all()

    connected: List[ConnectedWork] = []

    for row in cites_rows:
        connected.append(
            ConnectedWork(
                work_id=row["to_work_id"],
                title=row["title"],
                relationship="cites",
            )
        )

    for row in cited_by_rows:
        connected.append(
            ConnectedWork(
                work_id=row["from_work_id"],
                title=row["title"],
                relationship="cited_by",
            )
        )

    return connected


def _build_grounded_prompt(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    cited_by_count: int = 0,
    full_text_sections: Optional[Dict[str, str]] = None,
) -> str:
    """Build the LLM prompt with grounded paper context."""
    year_str = f" ({year})" if year else ""
    # Use full abstract — we have room in 128k context
    abstract_text = abstract or "No abstract available."

    # Detect potential pioneering work based on citation impact
    # High citation count is a strong signal of paradigm-shifting work
    # We provide context to help the LLM, but don't override its judgment
    paper_age = max(2025 - year, 1) if year else 1
    cites_per_year = cited_by_count / paper_age if cited_by_count > 0 else 0
    is_extremely_cited = cited_by_count > 50000
    is_old_and_foundational = (
        year is not None and year < 2000
        and cited_by_count > 5000
        and len(referenced_works) < 10
    )
    is_exceptional_rate = cites_per_year > 500 and len(referenced_works) < 10
    is_candidate_pioneer = is_extremely_cited or is_old_and_foundational or is_exceptional_rate
    pioneering_context = ""
    if is_candidate_pioneer:
        pioneering_context = f"""
## HIGH-IMPACT PAPER CONTEXT
This paper has {cited_by_count:,} citations, indicating significant field impact.
Consider whether this paper caused a PARADIGM SHIFT (pioneering) or made a significant contribution within existing paradigms (high).

Key question: Did the field fundamentally change how it operates AFTER this paper?
- If YES (before/after divide, everyone adopted this approach) → PIONEERING
- If NO (important but coexists with alternatives) → HIGH
"""

    # Build full text context block if available
    full_text_block = ""
    if full_text_sections:
        parts = []
        if full_text_sections.get("introduction"):
            parts.append(f"=== INTRODUCTION ===\n{full_text_sections['introduction']}")
        if full_text_sections.get("methods"):
            parts.append(f"=== METHODOLOGY ===\n{full_text_sections['methods']}")
        if full_text_sections.get("results_conclusion"):
            parts.append(f"=== RESULTS & CONCLUSIONS ===\n{full_text_sections['results_conclusion']}")
        if parts:
            full_text_block = "\n\nFULL TEXT SECTIONS (extracted from PDF — use these for detailed analysis):\n" + "\n\n".join(parts)

    # Build references section
    refs_section = ""
    if referenced_works:
        refs_lines = []
        for i, ref in enumerate(referenced_works[:10], 1):
            ref_title = ref.get("title") or "Untitled"
            ref_year = ref.get("year") or "?"
            ref_category = ref.get("category", "")
            category_label = f" [{ref_category}]" if ref_category else ""
            ref_abstract = _truncate_text(ref.get("abstract") or "", 600)
            refs_lines.append(
                f"{i}. [{ref.get('work_id')}]{category_label} {ref_title} ({ref_year})"
            )
            if ref_abstract:
                refs_lines.append(f"   Abstract: {ref_abstract}")
        refs_section = "\n".join(refs_lines)
    else:
        refs_section = "(No referenced works available)"

    # Build landmarks section with conditional instruction
    landmarks_section = ""
    landmark_instruction = ""

    if landmarks:
        landmark_lines = []
        for i, lm in enumerate(landmarks, 1):
            lm_title = lm.get("title") or "Untitled"
            lm_year = lm.get("year") or "?"
            lm_cites = lm.get("cited_by_count") or 0
            lm_category = lm.get("category", "")
            category_label = f" [{lm_category}]" if lm_category else ""
            lm_abstract = _truncate_text(lm.get("abstract") or "", 600)
            landmark_lines.append(
                f"{i}. [{lm.get('work_id')}]{category_label} {lm_title} ({lm_year}, {lm_cites:,} citations)"
            )
            if lm_abstract:
                landmark_lines.append(f"   Abstract: {lm_abstract}")
        landmarks_section = "\n".join(landmark_lines)
        landmark_instruction = "Compare to BOTH the references above AND these landmark papers."
    else:
        year_note = f" (target paper year: {year})" if year else ""
        landmarks_section = (
            f"(No landmark papers available - this may indicate an emerging topic "
            f"or the paper{year_note} is too recent for prior landmark identification)"
        )
        landmark_instruction = "Focus your comparison ONLY on the cited references above, as landmark context is unavailable."

    prompt = f"""Analyze this academic paper and provide a detailed assessment.

## Target Paper
Title: {title}{year_str}
Abstract: {abstract_text}
{full_text_block}
{pioneering_context}
## Papers This Work Cites (References)
{refs_section}

## Landmark Papers in This Field
{landmarks_section}

---

{landmark_instruction}

## NOVELTY CLASSIFICATION - ANSWER THESE IN ORDER:

**Q0: Is this a REVIEW, SYNTHESIS, META-ANALYSIS, TEXTBOOK, or CLINICAL MANUAL?**
- If title contains "review", "survey", "meta-analysis", "textbook", "handbook", "overview", "synthesis", "state of the art", "comprehensive", "progress in", "advances in", "perspectives on" → YES
- If title is "[Therapy/Treatment] for [Condition]" pattern (e.g., "Cognitive Therapy for Depression", "Behavior Therapy for X") → LIKELY a clinical manual/textbook
- If title is "[Technology/System] for [multiple applications]" pattern (e.g., "CRISPR-Cas systems for editing, regulating and targeting genomes") → LIKELY a review/overview
- CRITICAL: If title lists MULTIPLE applications/uses (contains "and" connecting 2+ applications), it's describing what something CAN do, not what the paper DOES → LIKELY a review
- If abstract mentions "comprehensive overview", "current state", "recent advances", "summarizes", "compiles", "practitioners", "clinicians", "treatment manual" → YES
- If paper has many references (>50) but abstract doesn't claim novel results → LIKELY a review
- If paper has 9000+ citations AND title matches "[Therapy] for [Condition]" pattern → LIKELY a seminal textbook/manual
- If YES → novelty_level = "medium" (STOP HERE - reviews/textbooks/manuals codify existing knowledge)

**EXCEPTION to Q0 - MAJOR GLOBAL ASSESSMENTS (can be HIGH):**
- If paper is a FIRST-OF-ITS-KIND global assessment that established new understanding → "high" (not medium)
- Example: First comprehensive global assessment of pollinator decline → "high" (changed policy)
- Example: IPCC climate assessments → "high" (synthesize but establish new policy-relevant findings)
- Key test: Did this assessment CHANGE how the field/policy thinks about the issue? → "high"
- Simple literature reviews that just compile what's known → "medium"

**Q1: Is the MAIN contribution a software tool, library, package, or data standard?**
- If title contains "library", "tool", "format", "software", "package" → YES
- If abstract describes implementing/providing software → YES
- If YES → novelty_level = "medium" (STOP HERE - no exceptions, even for highly cited software)

**Q2: Did this paper CREATE something new that fundamentally changed the field?**
(Only if Q1 = NO)

USE THE GROUNDING PAPERS AS EVIDENCE - look at the references/landmarks provided above:

**TEST: Do the grounding papers address the SAME TASK as the target paper?**

If YES (grounding papers work on the same task/problem):
- The task EXISTED before → this paper IMPROVED it → "high" at most
- Examples: better accuracy, faster speed, new architecture for same problem
- If prior papers do object detection and this paper does object detection → "high"
- If prior papers do image classification and this paper does image classification → "high"
- If prior papers do segmentation and this paper does segmentation → "high"

If NO (grounding papers work on FUNDAMENTALLY DIFFERENT tasks):
- Check: did this paper create a task that HAD NO PRIOR PAPERS attempting it?
- "pioneering" only if the APPLICATION itself is new (not just the method)
- Example: if no prior papers attempted artistic style transfer, and this paper created it → "pioneering"

PIONEERING is EXTREMELY RARE - requires:
1. A task/application that NOBODY was working on before
2. The grounding papers are from different domains being COMBINED into something new
3. NOT just a new method for an existing task

HIGH means:
- The task/problem already existed (grounding papers work on it)
- This paper provided a major improvement (new method, better results)

CRITICAL - Default to "high" unless evidence strongly supports "pioneering":
- Most influential papers are "high" (major improvements to existing tasks)
- "Pioneering" is RARE - reserved for papers that DEFINED new fields
- If ANY grounding paper addresses the same task → "high" not "pioneering"

CRITICAL - NOT high (these are "medium"):
- Systematizing or providing guidelines for an EXISTING method → "medium"
- Providing best practices or tutorials → "medium"
- Proposing better parameters/thresholds for existing methods → "medium"
- Creating a framework that UNIFIES existing methods without new capabilities → "medium"

CITATION COUNT IS NOT A NOVELTY INDICATOR:
- High citations mean IMPACT, not NOVELTY
- A highly-cited improvement to an existing task is "high", not "pioneering"

**THEORIES AND FRAMEWORKS:**
- Theory with NEW TESTABLE PREDICTIONS → "high"
- Theory that ORGANIZES existing knowledge → "medium"
- Framework that UNIFIES existing interpretation methods → "medium" (not pioneering)

**Q3: Did this paper significantly improve how we do something?**
(Only if Q1 = NO and Q2 did not result in pioneering)
- Major improvement that became widely adopted → "high"
- Incremental improvement or application → "medium"

**CRITICAL LANGUAGE RULES (READ FIRST):**
BANNED VERBS - NEVER use these in summary, whats_new, explanation, or relevance:
- explores, discusses, examines, investigates, assesses, evaluates, addresses, looks at, studies, analyzes, reviews
- If title uses "Assessing X" → you write "introduces/develops a method for X"
- If title uses "Investigating Y" → you write "establishes/demonstrates Y"

Return JSON (include q0_is_review, q1_is_software, q2_new_framework, q3_improvement to show your reasoning):
{{
    "summary": "MUST use: introduces/develops/demonstrates/establishes/creates/presents/validates. NEVER use banned verbs.",
    "keywords": ["keyword1", "keyword2", ...],
    "novelty_assessment": {{
        "q0_is_review": true | false,
        "q1_is_software": true | false,
        "q2_new_framework": "yes_pioneering" | "updates_existing" | "no",
        "q3_improvement": "major" | "incremental" | "n/a",
        "novelty_level": "low" | "medium" | "high" | "pioneering",
        "confidence": "low" | "medium" | "high",
        "whats_new": "NEVER use banned verbs. Use: introduces/builds upon/develops. Cite work_ids. ONLY null if pioneering. For reviews/surveys: describe what the review SYNTHESIZES or ORGANIZES (e.g. 'Synthesizes recent advances in X, organizing methods by Y'). NEVER null for reviews.",
        "compared_to_prior_work": "Compare to landmarks (cite work_ids). ONLY null if pioneering. For reviews: explain how it builds on prior work.",
        "novelty_explanation": "REQUIRED: (1) Cite specific work_ids [W...] that support classification, (2) Make specific technical claims about WHY this level. BAD: 'classified as review due to title'. GOOD: 'classified as high because it validates SOFA scores [W1898928487] for sepsis diagnosis, updating the SIRS criteria established by [W2168803832]'. NEVER just restate the Q0-Q3 decision without substantive technical claims.",
        "grounding_papers": [
            {{
                "work_id": "W...",
                "title": "...",
                "year": 2020,
                "cited_by_count": 1000,
                "relationship": "cited_reference" | "field_landmark",
                "relevance": "NEVER use banned verbs. Use: established/introduced/developed/demonstrated"
            }}
        ]
    }}
}}

**GROUNDING PAPERS:**
- Include 5-7 papers (mix of cited_reference + field_landmark)
- ONLY use work_ids from the lists provided above
- Each relevance MUST be SPECIFIC about technical relationship
- BAD: "Landmark paper in this field"
- GOOD: "Established SNe Ia as standard candles, which this paper uses to measure cosmic distances"
- Landmarks should be from the SAME FIELD (cosmology papers for cosmology, not particle physics textbooks)

**CRITICAL VALIDATION (CHECK YOUR OUTPUT BEFORE RETURNING):**
If your summary contains "examines", "discusses", "investigates", "likely", "assesses" → REWRITE IT
If your novelty_explanation says "Classified as X due to title/abstract" → REWRITE IT with technical claims

**WRITING QUALITY (applies to ALL output fields - summary, whats_new, explanation, relevance):**
- Be DEFINITIVE - state what the paper DOES and FINDS, not what it "explores" or "discusses"
- NEVER use hedging words: "likely", "might", "potentially", "possibly", "appears to", "seems to"
- ALL OUTPUT MUST state what the paper CREATES or FINDS, not what it "does" or "examines"
- DO NOT copy verbs from the title - rephrase completely
- If title says "Assessing X" → output says "develops/introduces a method/scale/instrument for X"
- If title says "Investigating Y" → output says "demonstrates/shows/establishes Y"
- FORBIDDEN verbs (NEVER use in ANY form - present, past, or gerund): explore/explored/exploring, discuss/discussed/discussing, examine/examined/examining, investigate/investigated/investigating, assess/assessed/assessing, address/addressed/addressing, look at/looked at/looking at, study/studied/studying, analyze/analyzed/analyzing, evaluate/evaluated/evaluating, review/reviewed/reviewing (as a verb)
- USE INSTEAD: introduce/introduced/introducing, develop/developed/developing, demonstrate/demonstrated/demonstrating, show/showed/showing, establish/established/establishing, create/created/creating, present/presented/presenting, propose/proposed/proposing, validate/validated/validating, build upon/built upon/building upon
- BAD summary: "This paper examines the importance of X" → GOOD: "This paper establishes X as critical to Y"
- BAD summary: "This paper discusses metals and toxicity" → GOOD: "This paper synthesizes evidence linking metal exposure to oxidative stress"
- BAD relevance: "Examines related topics" → GOOD: "Established the concept of X that this paper builds upon"
- If uncertain, lower your confidence level instead of hedging in the text

**NOVELTY_EXPLANATION REQUIREMENTS:**
- MUST cite at least 2 work_ids [W...] in the explanation
- MUST make specific technical claims (what method? what finding? what improvement?)
- FORBIDDEN patterns (will fail validation):
  - "Classified as X due to title containing..."
  - "Classified as review due to its synthesis..."
  - "The paper is X because it doesn't create..."
- REQUIRED pattern: "Classified as X because [specific technical contribution] builds upon [W...] and updates [specific prior work]"

**HANDLING MISSING ABSTRACTS:**
- If abstract says "No abstract available", you MUST still provide a meaningful summary
- CRITICAL: Do NOT echo the title. Completely rephrase using contribution-focused language.
- Template for missing-abstract summaries: "This paper [INTRODUCES/DEVELOPS/ESTABLISHES] [CONTRIBUTION], [OUTCOME/FINDING]."
- Example rewrites:
  - Title: "Assessing the quality of X" → Summary: "This paper introduces a validated scale for evaluating X quality"
  - Title: "Investigating whether Y" → Summary: "This paper establishes criteria for determining Y"
  - Title: "A study of Z" → Summary: "This paper demonstrates the relationship between Z components"
- Set confidence to "low" when abstract is missing"""

    return prompt


def generate_node_details_llm(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    target_work_id: Optional[str] = None,
    cited_by_count: int = 0,
    full_text_sections: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Call LLM to generate summary, keywords, and grounded novelty assessment."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, returning default response")
        return _default_response(title, referenced_works, landmarks)

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    prompt = _build_grounded_prompt(
        title, abstract, year, referenced_works, landmarks, cited_by_count,
        full_text_sections=full_text_sections,
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic paper analyst. Return only valid JSON. "
                        "Always cite specific papers by their work_id when making claims about prior work.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=90.0,
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Use robust JSON extraction (handles code blocks, extra text, etc.)
            result, error = extract_json_from_llm_response(content, expected_type="object")

            if result is None:
                logger.warning(f"Failed to parse LLM response JSON: {error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                    continue
                break

            return _validate_llm_response(result, referenced_works, landmarks, target_work_id, target_title=title)
        except Exception as e:
            logger.warning(f"LLM call exception: {e}")
            error_str = str(e).lower()
            is_transient = (
                "rate" in error_str
                or "timeout" in error_str
                or "connection" in error_str
            )
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                continue
            break

    return _default_response(title, referenced_works, landmarks)


def _extract_cited_work_ids(text: str) -> set[str]:
    """Extract work_ids mentioned in narrative text (format: W123456789)."""
    import re
    return set(re.findall(r'W\d{8,}', text))


def _check_grounding_consistency(
    novelty_assessment: Dict[str, Any],
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate that work_ids in narrative match grounding_papers.

    Returns consistency metrics including score and lists of mismatches.
    """
    # Extract all work_ids from narrative text (handle None values for pioneering works)
    whats_new = novelty_assessment.get("whats_new") or ""
    compared_to = novelty_assessment.get("compared_to_prior_work") or ""
    explanation = novelty_assessment.get("novelty_explanation") or ""
    narrative_text = f"{whats_new} {compared_to} {explanation}"
    cited_ids = _extract_cited_work_ids(narrative_text)

    # Get grounding_papers work_ids
    grounding_ids = {
        gp["work_id"]
        for gp in novelty_assessment.get("grounding_papers", [])
    }

    # Calculate consistency
    orphaned = cited_ids - grounding_ids  # Cited in text but not in grounding
    unused = grounding_ids - cited_ids    # In grounding but not mentioned

    consistency_score = 1.0
    if len(cited_ids) > 0:
        consistency_score = len(cited_ids & grounding_ids) / len(cited_ids)

    return {
        "consistency_score": consistency_score,
        "orphaned_citations": list(orphaned),
        "unused_grounding_papers": list(unused),
    }


def _filter_cross_domain_papers(
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    target_field_name: Optional[str] = None,
    target_topic_id: Optional[str] = None,
    target_title: Optional[str] = None,
    target_abstract: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter cross-domain papers from references and landmarks.

    ROOT CAUSE FIX: Uses METHODOLOGY-based filtering to catch papers
    that are in the same subfield but use different methods (e.g.,
    neural network papers for ensemble methods).

    FALLBACK: If strict filtering removes too many papers (OpenAlex sometimes
    misclassifies papers), keep the top-cited references anyway. Author-declared
    citations are inherently relevant.

    Args:
        referenced_works: Papers the target work cites
        landmarks: Field landmark papers
        target_field_name: OpenAlex subfield name (fallback if no topic_id)
        target_topic_id: OpenAlex topic ID of target paper (most precise filter)
        target_title: Title of target paper for methodology detection
        target_abstract: Abstract of target paper for methodology detection

    Returns:
        (filtered_refs, filtered_landmarks) with cross-domain papers removed
    """
    MIN_REFS_AFTER_FILTER = 3  # Keep at least this many refs if available

    # Use centralized methodology detection
    target_methodology = get_methodology(f"{target_title or ''} {target_abstract or ''}")
    if target_methodology:
        logger.info(f"Target methodology: {target_methodology} (from: {target_title[:50] if target_title else 'no title'}...)")

    def check_methodology_mismatch(paper: Dict[str, Any]) -> bool:
        """Check if paper uses a different methodology than target."""
        if is_methodology_mismatch(target_title, target_abstract, paper.get('title'), paper.get('abstract')):
            logger.info(
                f"FILTERING methodology mismatch: {paper.get('title', '')[:50]}..."
            )
            return True
        return False

    # If we don't know either the target field or topic, can't do subfield filtering
    # but we can still do methodology filtering
    if not target_field_name and not target_topic_id and not target_methodology:
        return referenced_works, landmarks

    # Cache for topic_id -> subfield lookups to avoid repeated API calls
    subfield_cache: Dict[str, Optional[str]] = {}

    def get_paper_subfield(paper: Dict[str, Any]) -> Optional[str]:
        """Get subfield for a paper, looking up from topic_id if needed."""
        # First check if field_name is already set
        if paper.get("field_name"):
            return paper["field_name"]

        # If paper has primary_topic_id, look up its subfield
        topic_id = paper.get("primary_topic_id")
        if topic_id:
            if topic_id not in subfield_cache:
                subfield_cache[topic_id] = get_field_from_topic_id(topic_id)
            return subfield_cache[topic_id]

        return None

    def is_cross_domain(paper: Dict[str, Any]) -> bool:
        work_id = paper.get("work_id", "")

        # FIRST: Check methodology mismatch (most precise filter)
        # This catches papers in the same subfield but different methods
        # e.g., neural network papers vs ensemble methods
        if check_methodology_mismatch(paper):
            return True

        # SECOND: Use SUBFIELD comparison as fallback
        # Topic IDs are too specific: ResNet vs DenseNet have different topics but are related
        # Subfield "Computer Vision and Pattern Recognition" correctly groups them
        paper_field = get_paper_subfield(paper)

        # If paper has subfield, compare directly
        if paper_field and target_field_name:
            if paper_field != target_field_name:
                logger.debug(
                    f"Cross-domain paper: {work_id} "
                    f"(field: {paper_field}, target: {target_field_name})"
                )
                return True
            return False

        # Paper has no subfield - either no topic_id or couldn't resolve
        # S2/ArXiv papers without OpenAlex field_name should be excluded
        if work_id.startswith("S2:") or work_id.startswith("ArXiv:"):
            logger.debug(
                f"Excluding unresolved S2/ArXiv paper: {work_id} "
                f"(no OpenAlex subfield for validation)"
            )
            return True

        # OpenAlex papers without topic_id (often very old papers)
        # Use content-word overlap as fallback — if no overlap, likely unrelated
        # Uses stop-word-free overlap so "in", "a", "the" don't create false matches
        paper_title = paper.get("title", "")
        if target_title and paper_title:
            overlap = content_word_overlap(target_title, paper_title)
            if overlap < 0.05:  # Near-zero content overlap = unrelated field
                logger.info(
                    f"Excluding unrelated old paper (overlap={overlap:.2f}): "
                    f"'{paper_title[:50]}...' vs target '{target_title[:50]}...'"
                )
                return True

        return False

    # First pass: strict subfield filtering
    filtered_refs = [r for r in referenced_works if not is_cross_domain(r)]
    filtered_landmarks = [lm for lm in landmarks if not is_cross_domain(lm)]

    # FALLBACK: If strict filtering removed too many references, OpenAlex may
    # have misclassified the target paper. Keep top-cited references anyway.
    # Author-declared citations (from OpenAlex referenced_works) are inherently
    # relevant - the author chose to cite them.
    if len(filtered_refs) < MIN_REFS_AFTER_FILTER and len(referenced_works) >= MIN_REFS_AFTER_FILTER:
        # Sort by citation count and take top refs
        sorted_refs = sorted(
            referenced_works,
            key=lambda x: x.get("cited_by_count", 0),
            reverse=True
        )
        # Only keep OpenAlex refs with SOME keyword overlap to target
        # (prevents re-adding clearly unrelated papers like Brownian Motion)
        openalex_refs = [
            r for r in sorted_refs
            if r.get("work_id", "").startswith("W")
            and (not target_title or content_word_overlap(target_title, r.get("title", "")) >= 0.05)
        ]
        if openalex_refs:
            filtered_refs = openalex_refs[:max(MIN_REFS_AFTER_FILTER, len(filtered_refs))]
            logger.info(
                f"Cross-domain fallback: strict filter left {len([r for r in referenced_works if not is_cross_domain(r)])} refs, "
                f"keeping {len(filtered_refs)} top-cited refs (OpenAlex misclassification likely)"
            )

    # Log filtering results
    ref_filtered = len(referenced_works) - len(filtered_refs)
    lm_filtered = len(landmarks) - len(filtered_landmarks)
    if ref_filtered > 0 or lm_filtered > 0:
        logger.info(
            f"Cross-domain filtering: refs {len(referenced_works)}->{len(filtered_refs)} "
            f"(-{ref_filtered}), landmarks {len(landmarks)}->{len(filtered_landmarks)} "
            f"(-{lm_filtered})"
        )

    return filtered_refs, filtered_landmarks


def _validate_llm_response(
    result: Dict[str, Any],
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    target_work_id: Optional[str] = None,
    target_title: str = "",
) -> Dict[str, Any]:
    """Validate and normalize the LLM response.

    Note: Cross-domain filtering happens BEFORE this function is called
    (in get_node_details via _filter_cross_domain_papers), so referenced_works
    and landmarks are already filtered.
    """
    summary = result.get("summary", "Summary unavailable")
    keywords = result.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    novelty = result.get("novelty_assessment", {})

    # Use already-filtered refs and landmarks (filtering happens earlier in the pipeline)
    # Methodology-based filtering is done in grounding_supplement.py
    filtered_refs = referenced_works
    filtered_landmarks = landmarks

    # Validate grounding papers
    grounding_papers = []
    raw_grounding = novelty.get("grounding_papers", [])
    valid_work_ids = {r["work_id"] for r in filtered_refs} | {
        lm["work_id"] for lm in filtered_landmarks
    }

    # Remove self-citation from valid work_ids (a paper should not cite itself as grounding)
    if target_work_id and target_work_id in valid_work_ids:
        valid_work_ids.discard(target_work_id)
        logger.info(f"Removed self-citation {target_work_id} from valid grounding papers")

    for gp in raw_grounding:
        if isinstance(gp, dict) and gp.get("work_id") in valid_work_ids:
            grounding_papers.append({
                "work_id": gp["work_id"],
                "title": gp.get("title", ""),
                "year": gp.get("year"),
                "cited_by_count": gp.get("cited_by_count"),
                "relationship": gp.get("relationship", "cited_reference"),
                "relevance": gp.get("relevance", ""),
            })

    # Ensure we have sufficient grounding papers (target: 5-7)
    # ALWAYS include at least 3 landmarks for robust field context
    MIN_GROUNDING = 5
    MAX_GROUNDING = 7
    MIN_LANDMARKS = 3

    grounding_work_ids = {gp["work_id"] for gp in grounding_papers}

    # Prevent self-citation: add target_work_id to grounding_work_ids so it's never added
    if target_work_id:
        grounding_work_ids.add(target_work_id)

    # Count current landmarks in grounding papers
    # Use the actual landmark work_ids, not the relationship string (more reliable)
    landmark_work_ids = {lm["work_id"] for lm in filtered_landmarks}
    current_landmark_count = sum(
        1 for gp in grounding_papers
        if gp["work_id"] in landmark_work_ids
    )

    # FIRST: Ensure we have at least MIN_LANDMARKS landmarks
    if current_landmark_count < MIN_LANDMARKS and filtered_landmarks:
        landmarks_needed = MIN_LANDMARKS - current_landmark_count
        sorted_landmarks = sorted(
            filtered_landmarks,
            key=lambda x: x.get("cited_by_count", 0) or 0,
            reverse=True
        )
        for lm in sorted_landmarks:
            if landmarks_needed <= 0:
                break
            if lm["work_id"] not in grounding_work_ids:
                grounding_papers.append({
                    "work_id": lm["work_id"],
                    "title": lm.get("title", ""),
                    "year": lm.get("year"),
                    "cited_by_count": lm.get("cited_by_count"),
                    "relationship": lm.get("relationship", "field_landmark"),
                    "relevance": _generate_relevance(lm, "field_landmark"),
                })
                grounding_work_ids.add(lm["work_id"])
                landmarks_needed -= 1

    # THEN: Add refs to reach minimum grounding count
    if len(grounding_papers) < MIN_GROUNDING:
        sorted_refs = sorted(
            filtered_refs,
            key=lambda x: x.get("cited_by_count", 0) or 0,
            reverse=True
        )
        for ref in sorted_refs:
            if len(grounding_papers) >= MAX_GROUNDING:
                break
            if ref["work_id"] not in grounding_work_ids:
                grounding_papers.append({
                    "work_id": ref["work_id"],
                    "title": ref.get("title", ""),
                    "year": ref.get("year"),
                    "cited_by_count": ref.get("cited_by_count"),
                    "relationship": ref.get("relationship", "cited_reference"),
                    "relevance": _generate_relevance(ref, "cited_reference"),
                })
                grounding_work_ids.add(ref["work_id"])

    # FINALLY: Add more landmarks if still under minimum
    if len(grounding_papers) < MIN_GROUNDING:
        sorted_landmarks = sorted(
            filtered_landmarks,
            key=lambda x: x.get("cited_by_count", 0) or 0,
            reverse=True
        )
        for lm in sorted_landmarks:
            if len(grounding_papers) >= MAX_GROUNDING:
                break
            if lm["work_id"] not in grounding_work_ids:
                grounding_papers.append({
                    "work_id": lm["work_id"],
                    "title": lm.get("title", ""),
                    "year": lm.get("year"),
                    "cited_by_count": lm.get("cited_by_count"),
                    "relationship": lm.get("relationship", "field_landmark"),
                    "relevance": _generate_relevance(lm, "field_landmark"),
                })
                grounding_work_ids.add(lm["work_id"])

    # Log if still under minimum (indicates data issue)
    # Do NOT force add papers - better to have fewer than cross-domain contamination
    if len(grounding_papers) < MIN_GROUNDING:
        logger.warning(
            f"Validation could not reach {MIN_GROUNDING} grounding papers. "
            f"Have {len(grounding_papers)}, filtered_refs={len(filtered_refs)}, filtered_landmarks={len(filtered_landmarks)}. "
            f"Accepting fewer to avoid cross-domain contamination."
        )

    # POST-ASSEMBLY FILTER: Remove grounding papers with boilerplate relevance
    # that have no real connection to the target paper.
    # Uses content_word_overlap (stop words removed) for accurate topical comparison.
    # Two tiers:
    #   - Zero content overlap + boilerplate → ALWAYS remove (clearly wrong, e.g., Brownian Motion)
    #   - Low content overlap + boilerplate → remove only if above minimum count
    if target_title:
        clearly_irrelevant = []
        borderline_boilerplate = []
        real_papers = []
        for gp in grounding_papers:
            relevance = gp.get("relevance", "")
            gp_title = gp.get("title", "")
            is_boilerplate = (
                relevance.startswith("Cited by this paper for its contribution:")
                or relevance.startswith("Contributed key advances (")
                or relevance.startswith("Introduced early methods (")
            )
            if is_boilerplate:
                overlap = content_word_overlap(target_title, gp_title) if gp_title else 0
                if overlap < 0.05:
                    # Zero content overlap + boilerplate = clearly wrong field, always remove
                    clearly_irrelevant.append(gp)
                    logger.info(
                        f"Removing clearly irrelevant grounding paper (content_overlap={overlap:.3f}): "
                        f"'{gp_title[:60]}'"
                    )
                    continue
                elif overlap < 0.10:
                    # Low content overlap + boilerplate = borderline, remove if above minimum
                    borderline_boilerplate.append(gp)
                    continue
            real_papers.append(gp)

        # Always remove clearly irrelevant (even if it drops below minimum)
        grounding_papers = real_papers + borderline_boilerplate

        # Remove borderline boilerplate if we'd still have enough
        if len(real_papers) >= MIN_GROUNDING:
            grounding_papers = real_papers
            if borderline_boilerplate:
                logger.info(f"Also removed {len(borderline_boilerplate)} borderline boilerplate papers")
        if clearly_irrelevant:
            logger.info(f"Removed {len(clearly_irrelevant)} clearly irrelevant grounding papers")

    # Handle None values for pioneering works (LLM may return null for whats_new/compared_to_prior_work)
    novelty_assessment = {
        "whats_new": novelty.get("whats_new") or None,  # Keep None for pioneering works
        "compared_to_prior_work": novelty.get("compared_to_prior_work") or None,  # Keep None for pioneering works
        "novelty_level": novelty.get("novelty_level") or "medium",
        "confidence": novelty.get("confidence") or "low",
        "novelty_explanation": novelty.get("novelty_explanation") or "Assessment unavailable",
        "grounding_papers": grounding_papers[:MAX_GROUNDING],
    }

    # Validate enum values
    if novelty_assessment["novelty_level"] not in ("low", "medium", "high", "pioneering"):
        novelty_assessment["novelty_level"] = "medium"
    if novelty_assessment["confidence"] not in ("low", "medium", "high"):
        novelty_assessment["confidence"] = "low"

    # Check grounding consistency
    consistency = _check_grounding_consistency(
        novelty_assessment,
        filtered_refs,
        filtered_landmarks,
    )

    # FIX orphaned citations: add work_ids mentioned in narrative but missing from grounding_papers
    orphaned = consistency.get("orphaned_citations", [])
    if orphaned and len(grounding_papers) < MAX_GROUNDING:
        all_papers_lookup = {r["work_id"]: r for r in filtered_refs}
        all_papers_lookup.update({lm["work_id"]: lm for lm in filtered_landmarks})
        landmark_ids = {lm["work_id"] for lm in filtered_landmarks}

        for orphan_id in orphaned:
            if len(grounding_papers) >= MAX_GROUNDING:
                break
            if orphan_id in all_papers_lookup and orphan_id not in grounding_work_ids:
                paper = all_papers_lookup[orphan_id]
                # Determine relationship based on source
                relationship = "field_landmark" if orphan_id in landmark_ids else "cited_reference"
                grounding_papers.append({
                    "work_id": orphan_id,
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "cited_by_count": paper.get("cited_by_count"),
                    "relationship": relationship,
                    "relevance": _generate_relevance(paper, relationship),
                })
                grounding_work_ids.add(orphan_id)
                logger.info(f"Added orphaned citation {orphan_id} to grounding_papers")

        # Recalculate consistency after fixing orphans
        consistency = _check_grounding_consistency(
            novelty_assessment,
            filtered_refs,
            filtered_landmarks,
        )

    # Log inconsistencies
    if consistency["consistency_score"] < 0.8:
        logger.warning(
            f"Low consistency score: {consistency['consistency_score']:.2f}",
            extra={
                "consistency_score": consistency["consistency_score"],
                "orphaned_citations": consistency["orphaned_citations"],
                "unused_grounding_papers": consistency["unused_grounding_papers"],
            }
        )

    # Reduce confidence if consistency is very low
    if consistency["consistency_score"] < 0.5 and novelty_assessment["confidence"] == "high":
        novelty_assessment["confidence"] = "medium"
        logger.info("Reduced confidence from high to medium due to low grounding consistency")

    # ================================================================
    # POST-LLM VALIDATION SWEEP (code-level guardrails for durability)
    # ================================================================

    # 1. Scrub banned verbs from ALL text fields
    summary = _scrub_banned_verbs(summary)
    for field_name in ("whats_new", "compared_to_prior_work", "novelty_explanation"):
        val = novelty_assessment.get(field_name)
        if val:
            novelty_assessment[field_name] = _scrub_banned_verbs(val)

    # Scrub banned verbs from grounding paper relevance strings
    for gp in novelty_assessment.get("grounding_papers", []):
        if gp.get("relevance"):
            gp["relevance"] = _scrub_banned_verbs(gp["relevance"])

    # 2. Fix whats_new=null for non-pioneering papers
    #    Rubric says "ONLY null if pioneering"
    level = novelty_assessment.get("novelty_level", "medium")
    if level != "pioneering" and not novelty_assessment.get("whats_new"):
        # For reviews (medium via Q0), describe what the review synthesizes
        compared = novelty_assessment.get("compared_to_prior_work") or ""
        explanation = novelty_assessment.get("novelty_explanation") or ""
        if "review" in explanation.lower() or "synthes" in explanation.lower():
            novelty_assessment["whats_new"] = (
                f"Synthesizes and organizes current knowledge in this area, "
                f"building upon prior work cited in the assessment."
            )
        else:
            # Non-review: derive from compared_to_prior_work or explanation
            if compared:
                novelty_assessment["whats_new"] = compared[:200]
            elif explanation:
                novelty_assessment["whats_new"] = explanation[:200]
        logger.info(f"Fixed null whats_new for non-pioneering paper (level={level})")

    # 3. Replace any remaining boilerplate relevance strings from LLM output
    for gp in novelty_assessment.get("grounding_papers", []):
        if gp.get("relevance") in _BOILERPLATE_RELEVANCE:
            gp["relevance"] = _generate_relevance(gp, gp.get("relationship", "cited_reference"))

    return {
        "summary": summary,
        "keywords": keywords[:10],
        "novelty_assessment": novelty_assessment,
        "_consistency_metrics": consistency,  # Internal use for monitoring
    }


def _build_impact_analysis_obj(impact_data: Optional[Dict[str, Any]]) -> Optional[PaperImpactAnalysis]:
    """Convert impact_analysis dict to PaperImpactAnalysis Pydantic model."""
    if not impact_data:
        return None
    return PaperImpactAnalysis(
        is_paradigm_shift=impact_data.get("is_paradigm_shift", False),
        impact_score=impact_data.get("impact_score", 0.0),
        before_approach=impact_data.get("before_approach"),
        after_approach=impact_data.get("after_approach"),
        shift_description=impact_data.get("shift_description"),
    )


def _default_response(
    title: str,
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a default response when LLM fails."""
    grounding_papers = []

    if referenced_works:
        top_ref = referenced_works[0]
        grounding_papers.append({
            "work_id": top_ref["work_id"],
            "title": top_ref.get("title", ""),
            "year": top_ref.get("year"),
            "cited_by_count": top_ref.get("cited_by_count"),
            "relationship": "cited_reference",
            "relevance": "Primary reference",
        })

    if landmarks:
        top_lm = landmarks[0]
        grounding_papers.append({
            "work_id": top_lm["work_id"],
            "title": top_lm.get("title", ""),
            "year": top_lm.get("year"),
            "cited_by_count": top_lm.get("cited_by_count"),
            "relationship": "field_landmark",
            "relevance": "Key landmark in this field",
        })

    return {
        "summary": f"Summary for: {title}",
        "keywords": [],
        "novelty_assessment": {
            "whats_new": "Unable to assess",
            "compared_to_prior_work": "Unable to assess",
            "novelty_level": "medium",
            "confidence": "low",
            "novelty_explanation": "Novelty assessment unavailable due to processing error.",
            "grounding_papers": grounding_papers,
        },
    }


def get_node_details(
    engine: Engine,
    *,
    tenant_id: UUID,
    map_id: UUID,
    work_id: str,
    include_novelty: bool = False,
    include_timeline: bool = False,
) -> NodeDetailsResponse:
    """
    Get detailed pop-up information for a node in the citation map.

    This is the main entry point for Feature 3.

    When include_novelty=False (default), returns lightweight metadata only
    (no LLM call). When True, runs the full novelty assessment pipeline.
    """
    with engine.connect() as conn:
        # Verify map exists and belongs to tenant
        map_row = conn.execute(
            text("""
                SELECT map_id FROM maps
                WHERE map_id = :map_id AND tenant_id = :tenant_id
            """),
            {"map_id": map_id, "tenant_id": tenant_id},
        ).first()

        if not map_row:
            raise ValueError("map_not_found")

        # Verify work is a node in this map
        node_row = conn.execute(
            text("""
                SELECT work_id FROM map_nodes
                WHERE map_id = :map_id AND work_id = :work_id
            """),
            {"map_id": map_id, "work_id": work_id},
        ).first()

        if not node_row:
            raise ValueError("node_not_found")

        # Load work metadata
        work_data = load_work_data(conn, work_id)
        if not work_data:
            raise ValueError("work_not_found")

        # Resolve access links
        from app.settings.access_links import resolve_access_link
        from app.settings.store import load_tenant_settings
        tenant_settings = load_tenant_settings(conn, tenant_id)
        access_info = resolve_access_link(
            doi=work_data.get("doi"),
            is_open_access=work_data.get("is_open_access"),
            oa_pdf_url=work_data.get("oa_pdf_url"),
            proxy_prefix=tenant_settings.get("institutional_proxy_prefix"),
            libkey_api_key=tenant_settings.get("libkey_api_key"),
            libkey_library_id=tenant_settings.get("libkey_library_id"),
        )

        # Enrich abstract if invalid (fetch from ArXiv/Semantic Scholar)
        enrichment_happened = False
        enriched_abstract, abstract_source = ensure_valid_abstract(
            conn,
            work_id=work_id,
            title=work_data["title"],
            abstract=work_data["abstract"],
            year=work_data["year"],
            doi=work_data.get("doi"),
            arxiv_id=work_data.get("arxiv_id"),
        )
        if abstract_source not in ("cached", "unavailable") and enriched_abstract:
            # Only flag enrichment if the abstract actually changed
            if enriched_abstract != work_data["abstract"]:
                logger.info(f"Enriched abstract for {work_id} from {abstract_source}")
                work_data["abstract"] = enriched_abstract
                enrichment_happened = True
        elif abstract_source == "unavailable":
            # Abstract was invalid and no replacement found - clear it so LLM knows
            if work_data["abstract"] is not None:
                logger.info(f"Abstract for {work_id} is invalid and unfetchable, clearing")
                work_data["abstract"] = None
                enrichment_happened = True

        # Infer topic if missing (needed for landmark retrieval)
        inferred_topic_id, topic_source = ensure_topic(
            conn,
            work_id=work_id,
            title=work_data["title"],
            abstract=work_data["abstract"],
            current_topic_id=work_data.get("primary_topic_id"),
            doi=work_data.get("doi"),
            arxiv_id=work_data.get("arxiv_id"),
        )
        if topic_source not in ("cached", "unavailable") and inferred_topic_id:
            # Only flag enrichment if the topic actually changed
            if inferred_topic_id != work_data.get("primary_topic_id"):
                logger.info(f"Inferred topic for {work_id}: {inferred_topic_id} from {topic_source}")
                work_data["primary_topic_id"] = inferred_topic_id
                enrichment_happened = True

        # Fetch topic display name for "Research this topic" feature
        topic_display_name = None
        if work_data.get("primary_topic_id"):
            topic_display_name = get_topic_display_name(conn, work_data["primary_topic_id"])
            if topic_display_name:
                logger.debug(f"Topic display name for {work_id}: {topic_display_name}")

        # Load connected works (from map edges)
        connected_works = load_connected_works(conn, map_id, work_id)

        # Abstract is REQUIRED for novelty assessment — LLM produces garbage without it
        if include_novelty and not work_data.get("abstract"):
            logger.warning(f"No abstract available for {work_id} - novelty assessment unavailable")
            basic_summary = _generate_summary_from_abstract(None, work_data["title"])
            basic_keywords = _extract_keywords_from_abstract(None)
            unavailable_reason = (
                "No abstract available for this paper. Novelty assessment requires "
                "at least an abstract to provide accurate analysis."
            )
            cache_details(
                conn, work_id, basic_summary, basic_keywords, None,
                assessment_unavailable_reason=unavailable_reason,
            )
            return NodeDetailsResponse(
                work_id=work_data["work_id"],
                title=work_data["title"],
                year=work_data["year"],
                authors=work_data["authors"],
                venue=work_data["venue"],
                cited_by_count=work_data["cited_by_count"],
                abstract=None,
                summary=basic_summary,
                keywords=basic_keywords,
                novelty_assessment=None,
                connected_works=connected_works,
                assessment_unavailable_reason=unavailable_reason,
                timeline=None,
                primary_topic_id=work_data.get("primary_topic_id"),
                topic_display_name=topic_display_name,
                access_status=access_info["access_status"],
                pdf_url=access_info["pdf_url"],
                doi_url=access_info["doi_url"],
                oa_status=work_data.get("oa_status"),
            )

        # Lightweight metadata-only response (no LLM call)
        if not include_novelty:
            basic_summary = _generate_summary_from_abstract(
                work_data["abstract"], work_data["title"]
            )
            basic_keywords = _extract_keywords_from_abstract(work_data["abstract"])

            return NodeDetailsResponse(
                work_id=work_data["work_id"],
                title=work_data["title"],
                year=work_data["year"],
                authors=work_data["authors"],
                venue=work_data["venue"],
                cited_by_count=work_data["cited_by_count"],
                abstract=work_data["abstract"],
                summary=basic_summary,
                keywords=basic_keywords,
                novelty_assessment=None,
                connected_works=connected_works,
                timeline=None,
                primary_topic_id=work_data.get("primary_topic_id"),
                topic_display_name=topic_display_name,
                access_status=access_info["access_status"],
                pdf_url=access_info["pdf_url"],
                doi_url=access_info["doi_url"],
                oa_status=work_data.get("oa_status"),
            )

        # Check cache first (but skip if enrichment happened - regenerate assessment)
        cached = None
        if not enrichment_happened:
            cached = get_cached_details(conn, work_id)

        if cached:
            logger.info(f"Cache hit for node details: {work_id}")

            # Handle case where novelty_assessment was not available
            novelty_assessment_obj = None
            if cached["novelty_assessment"]:
                # Extract grounding_papers separately to avoid duplicate argument
                novelty_dict = dict(cached["novelty_assessment"])
                grounding_papers_data = novelty_dict.pop("grounding_papers", [])
                novelty_assessment_obj = NoveltyAssessment(
                    **novelty_dict,
                    grounding_papers=[
                        GroundingPaper(**gp)
                        for gp in grounding_papers_data
                    ],
                )

            # Build timeline if requested (requires loading refs/landmarks)
            timeline_obj = None
            if include_timeline:
                logger.info(f"Building timeline for cached {work_id}")
                # Load references and landmarks for timeline
                referenced_works = get_referenced_works(conn, work_id)
                landmarks = get_topic_landmarks(
                    conn,
                    work_data.get("primary_topic_id"),
                    work_data.get("year"),
                )
                timeline_data = build_node_timeline(
                    conn,
                    work_id,
                    work_data["year"],
                    referenced_works,
                    landmarks,
                    include_impact_analysis=True,
                    target_title=work_data.get("title"),
                    target_abstract=work_data.get("abstract"),
                    target_cited_by_count=work_data.get("cited_by_count", 0),
                )
                timeline_obj = NodeTimeline(
                    target_work_id=timeline_data["target_work_id"],
                    target_year=timeline_data["target_year"],
                    backward=[
                        TimelineSection(
                            era=section["era"],
                            papers=[TimelinePaper(**p) for p in section["papers"]],
                        )
                        for section in timeline_data["backward"]
                    ],
                    forward=[
                        TimelineSection(
                            era=section["era"],
                            papers=[TimelinePaper(**p) for p in section["papers"]],
                        )
                        for section in timeline_data["forward"]
                    ],
                    impact_analysis=_build_impact_analysis_obj(timeline_data.get("impact_analysis")),
                )

            return NodeDetailsResponse(
                work_id=work_data["work_id"],
                title=work_data["title"],
                year=work_data["year"],
                authors=work_data["authors"],
                venue=work_data["venue"],
                cited_by_count=work_data["cited_by_count"],
                abstract=work_data["abstract"],
                summary=cached["summary"],
                keywords=cached["keywords"],
                novelty_assessment=novelty_assessment_obj,
                connected_works=connected_works,
                assessment_unavailable_reason=cached.get("assessment_unavailable_reason"),
                timeline=timeline_obj,
                primary_topic_id=work_data.get("primary_topic_id"),
                topic_display_name=topic_display_name,
                access_status=access_info["access_status"],
                pdf_url=access_info["pdf_url"],
                doi_url=access_info["doi_url"],
                oa_status=work_data.get("oa_status"),
            )

        # Fetch grounding papers
        logger.info(f"Fetching grounding papers for: {work_id}")

        # Get referenced works (papers this work cites)
        referenced_works = get_referenced_works(conn, work_id)
        logger.info(f"Found {len(referenced_works)} referenced works")

        # Get topic landmarks
        landmarks = get_topic_landmarks(
            conn,
            work_data["primary_topic_id"],
            work_data["year"],
        )
        logger.info(f"Found {len(landmarks)} topic landmarks")

        # Detect if this is a pioneering work (highly cited foundational paper)
        # Multiple signals: extreme citations, old + well-cited + few refs, high cite-per-year ratio
        cited_by_count = work_data.get("cited_by_count", 0)
        work_year = work_data.get("year")
        is_pioneering = False
        if work_year is not None and cited_by_count > 0:
            paper_age = max(2025 - work_year, 1)
            cites_per_year = cited_by_count / paper_age
            # Extremely cited papers (>50K) are always candidates
            if cited_by_count > 50000:
                is_pioneering = True
            # Old papers (pre-2000) with high citations AND few references (foundational work)
            elif work_year < 2000 and cited_by_count > 5000 and len(referenced_works) < 10:
                is_pioneering = True
            # Any paper with exceptional cite-per-year ratio AND few references
            elif cites_per_year > 500 and len(referenced_works) < 10:
                is_pioneering = True
        if is_pioneering:
            logger.info(f"Detected pioneering work: {work_id} (year={work_year}, citations={cited_by_count})")

        # Check if we have NO grounding data at all - try LLM landmark fallback
        # Only attempt LLM fallback for papers with >500 citations (significant works)
        if len(referenced_works) == 0 and len(landmarks) == 0:
            if cited_by_count > 500:
                logger.info(
                    f"No grounding data for {work_id} ({cited_by_count} citations) - "
                    f"trying LLM landmark fallback"
                )

                # Try to get landmarks via LLM suggestion (fallback for 0+0 case)
                _, fallback_landmarks = supplement_grounding_papers(
                    title=work_data["title"],
                    abstract=work_data["abstract"],
                    year=work_data["year"],
                    existing_refs=[],
                    existing_landmarks=[],
                    field=work_data.get("category"),
                    primary_topic_id=work_data.get("primary_topic_id"),
                    is_pioneering=is_pioneering,
                    target_work_id=work_id,
                    conn=conn,
                )
            else:
                logger.info(
                    f"No grounding data for {work_id} ({cited_by_count} citations) - "
                    f"skipping LLM fallback (requires >500 citations)"
                )
                fallback_landmarks = []

            if fallback_landmarks:
                logger.info(f"LLM fallback found {len(fallback_landmarks)} landmarks")
                landmarks = fallback_landmarks
            else:
                # Still no grounding - return UNAVAILABLE
                logger.warning(
                    f"No grounding data available for {work_id} - skipping novelty assessment"
                )

                basic_summary = _generate_summary_from_abstract(
                    work_data["abstract"], work_data["title"]
                )
                basic_keywords = _extract_keywords_from_abstract(work_data["abstract"])

                # Build timeline if requested (even without grounding data)
                timeline_obj = None
                if include_timeline:
                    logger.info(f"Building timeline for {work_id} (no grounding data)")
                    timeline_data = build_node_timeline(
                        conn,
                        work_id,
                        work_data["year"],
                        [],  # No references
                        [],  # No landmarks
                        include_impact_analysis=True,
                        target_title=work_data.get("title"),
                        target_abstract=work_data.get("abstract"),
                        target_cited_by_count=work_data.get("cited_by_count", 0),
                    )
                    timeline_obj = NodeTimeline(
                        target_work_id=timeline_data["target_work_id"],
                        target_year=timeline_data["target_year"],
                        backward=[
                            TimelineSection(
                                era=section["era"],
                                papers=[TimelinePaper(**p) for p in section["papers"]],
                            )
                            for section in timeline_data["backward"]
                        ],
                        forward=[
                            TimelineSection(
                                era=section["era"],
                                papers=[TimelinePaper(**p) for p in section["papers"]],
                            )
                            for section in timeline_data["forward"]
                        ],
                        impact_analysis=_build_impact_analysis_obj(timeline_data.get("impact_analysis")),
                    )

                unavailable_reason = (
                    "Insufficient reference data - no citations or field landmark papers "
                    "available for comparison. Novelty assessment requires at least one "
                    "reference or landmark paper to ground the analysis."
                )

                # Cache the unavailable result to avoid repeated expensive lookups
                cache_details(
                    conn, work_id,
                    basic_summary, basic_keywords,
                    None,
                    assessment_unavailable_reason=unavailable_reason,
                )

                return NodeDetailsResponse(
                    work_id=work_data["work_id"],
                    title=work_data["title"],
                    year=work_data["year"],
                    authors=work_data["authors"],
                    venue=work_data["venue"],
                    cited_by_count=work_data["cited_by_count"],
                    abstract=work_data["abstract"],
                    summary=basic_summary,
                    keywords=basic_keywords,
                    novelty_assessment=None,
                    connected_works=connected_works,
                    assessment_unavailable_reason=unavailable_reason,
                    timeline=timeline_obj,
                    primary_topic_id=work_data.get("primary_topic_id"),
                    topic_display_name=topic_display_name,
                    access_status=access_info["access_status"],
                    pdf_url=access_info["pdf_url"],
                    doi_url=access_info["doi_url"],
                    oa_status=work_data.get("oa_status"),
                )

        # ROOT CAUSE FIX: Get target paper's topic_id for precise cross-domain filtering
        # Subfields like "Artificial Intelligence" are too coarse (NLP + gradient boosting)
        # Topic IDs are specific (e.g., "Word Embeddings" vs "Gradient Boosting")
        target_topic_id = work_data.get("primary_topic_id")
        target_field_name = get_field_from_topic_id(target_topic_id)
        if target_topic_id:
            logger.info(f"Target paper topic: {target_topic_id}, field: {target_field_name}")

        # Pre-filter cross-domain papers BEFORE checking if supplement is needed
        # This ensures supplement sees the actual usable count and adds more if needed
        filtered_refs, filtered_landmarks = _filter_cross_domain_papers(
            referenced_works, landmarks, target_field_name, target_topic_id,
            target_title=work_data.get("title"),
            target_abstract=work_data.get("abstract"),
        )
        logger.info(
            f"Pre-supplement cross-domain filter: refs {len(referenced_works)}->{len(filtered_refs)}, "
            f"landmarks {len(landmarks)}->{len(filtered_landmarks)}"
        )

        # Supplement grounding papers if we have some but not enough (target: 5-7)
        additional_refs, additional_landmarks = supplement_grounding_papers(
            title=work_data["title"],
            abstract=work_data["abstract"],
            year=work_data["year"],
            existing_refs=filtered_refs,  # Use filtered versions
            existing_landmarks=filtered_landmarks,  # Use filtered versions
            field=work_data.get("category"),  # Use category as field hint
            primary_topic_id=work_data.get("primary_topic_id"),
            is_pioneering=is_pioneering,
            target_work_id=work_id,
            conn=conn,
        )

        if additional_refs or additional_landmarks:
            logger.info(
                f"Supplemented grounding: +{len(additional_refs)} refs, "
                f"+{len(additional_landmarks)} landmarks"
            )
            # Combine and RE-FILTER supplemented papers for methodology mismatch
            # Supplement uses subfield filtering but not methodology filtering
            combined_refs = filtered_refs + additional_refs
            combined_landmarks = filtered_landmarks + additional_landmarks
            filtered_refs, filtered_landmarks = _filter_cross_domain_papers(
                combined_refs, combined_landmarks, target_field_name, target_topic_id,
                target_title=work_data.get("title"),
                target_abstract=work_data.get("abstract"),
            )
            logger.info(
                f"Post-supplement methodology filter: refs {len(combined_refs)}->{len(filtered_refs)}, "
                f"landmarks {len(combined_landmarks)}->{len(filtered_landmarks)}"
            )

        # Use filtered versions from here on
        referenced_works = filtered_refs
        landmarks = filtered_landmarks

        # Attempt to fetch full text for richer novelty assessment
        full_text_sections = None
        context_depth = "abstract_only"

        try:
            cached_ft = conn.execute(
                text("SELECT methods_text, full_text_available FROM paper_full_text_cache WHERE work_id = :wid"),
                {"wid": work_id},
            ).mappings().first()

            if cached_ft and cached_ft["full_text_available"] and cached_ft["methods_text"]:
                full_text_sections = extract_paper_sections(cached_ft["methods_text"])
                context_depth = "full_text"
                logger.info(f"Using cached full text for {work_id}")
            else:
                # Try to download PDF
                pdf_url = work_data.get("oa_pdf_url")
                if not pdf_url and work_data.get("arxiv_id"):
                    pdf_url = f"https://arxiv.org/pdf/{work_data['arxiv_id']}.pdf"

                if pdf_url:
                    logger.info(f"Attempting PDF download for novelty assessment: {pdf_url[:80]}")
                    full_text = download_and_extract_pdf(pdf_url)
                    if full_text:
                        full_text_sections = extract_paper_sections(full_text)
                        context_depth = "full_text"
                        logger.info(f"Full text extracted for {work_id}: {sum(len(v) for v in full_text_sections.values())} chars")
                        # Cache for future use
                        try:
                            conn.execute(
                                text("""
                                    INSERT INTO paper_full_text_cache (work_id, methods_text, full_text_available, fetched_at)
                                    VALUES (:wid, :text, true, now())
                                    ON CONFLICT (work_id) DO UPDATE SET
                                        methods_text = EXCLUDED.methods_text,
                                        full_text_available = true,
                                        fetched_at = now()
                                """),
                                {"wid": work_id, "text": full_text[:50000]},
                            )
                            conn.commit()
                        except Exception as e:
                            logger.warning(f"Failed to cache full text for {work_id}: {e}")
        except Exception as e:
            logger.warning(f"Full text retrieval failed for {work_id}: {e}")

        # Generate via LLM with grounded context
        logger.info(f"Generating grounded node details via LLM for: {work_id}")
        llm_result = generate_node_details_llm(
            title=work_data["title"] or "Untitled",
            abstract=work_data["abstract"],
            year=work_data["year"],
            referenced_works=referenced_works,
            landmarks=landmarks,
            target_work_id=work_id,  # Prevent self-citation in grounding papers
            cited_by_count=work_data.get("cited_by_count", 0),
            full_text_sections=full_text_sections,
        )

        # Log quality metrics
        consistency_metrics = llm_result.get("_consistency_metrics", {})
        logger.info(
            "Generated novelty assessment",
            extra={
                "work_id": work_id,
                "reference_count": len(referenced_works),
                "landmark_count": len(landmarks),
                "grounding_paper_count": len(llm_result["novelty_assessment"].get("grounding_papers", [])),
                "consistency_score": consistency_metrics.get("consistency_score", 1.0),
                "novelty_level": llm_result["novelty_assessment"].get("novelty_level"),
                "confidence": llm_result["novelty_assessment"].get("confidence"),
                "has_landmark_gap": len(landmarks) < 3,
            }
        )

        # Cache the result
        cache_details(
            conn,
            work_id,
            llm_result["summary"],
            llm_result["keywords"],
            llm_result["novelty_assessment"],
        )

        # Build timeline if requested
        timeline_obj = None
        if include_timeline:
            logger.info(f"Building timeline for {work_id}")
            timeline_data = build_node_timeline(
                conn,
                work_id,
                work_data["year"],
                referenced_works,
                landmarks,
                include_impact_analysis=True,
                target_title=work_data.get("title"),
                target_abstract=work_data.get("abstract"),
                target_cited_by_count=work_data.get("cited_by_count", 0),
            )
            timeline_obj = NodeTimeline(
                target_work_id=timeline_data["target_work_id"],
                target_year=timeline_data["target_year"],
                backward=[
                    TimelineSection(
                        era=section["era"],
                        papers=[TimelinePaper(**p) for p in section["papers"]],
                    )
                    for section in timeline_data["backward"]
                ],
                forward=[
                    TimelineSection(
                        era=section["era"],
                        papers=[TimelinePaper(**p) for p in section["papers"]],
                    )
                    for section in timeline_data["forward"]
                ],
                impact_analysis=_build_impact_analysis_obj(timeline_data.get("impact_analysis")),
            )

        return NodeDetailsResponse(
            work_id=work_data["work_id"],
            title=work_data["title"],
            year=work_data["year"],
            authors=work_data["authors"],
            venue=work_data["venue"],
            cited_by_count=work_data["cited_by_count"],
            abstract=work_data["abstract"],
            summary=llm_result["summary"],
            keywords=llm_result["keywords"],
            novelty_assessment=NoveltyAssessment(
                whats_new=llm_result["novelty_assessment"]["whats_new"],
                compared_to_prior_work=llm_result["novelty_assessment"]["compared_to_prior_work"],
                novelty_level=llm_result["novelty_assessment"]["novelty_level"],
                confidence=llm_result["novelty_assessment"]["confidence"],
                novelty_explanation=llm_result["novelty_assessment"]["novelty_explanation"],
                grounding_papers=[
                    GroundingPaper(**gp)
                    for gp in llm_result["novelty_assessment"]["grounding_papers"]
                ],
                context_depth=context_depth,
            ),
            connected_works=connected_works,
            timeline=timeline_obj,
            primary_topic_id=work_data.get("primary_topic_id"),
            topic_display_name=topic_display_name,
            access_status=access_info["access_status"],
            pdf_url=access_info["pdf_url"],
            doi_url=access_info["doi_url"],
            oa_status=work_data.get("oa_status"),
        )


def get_novelty_for_work(
    engine: Engine,
    *,
    work_id: str,
) -> dict:
    """
    Run novelty assessment for a work_id without requiring a citation map.

    Returns a dict with:
        "novelty_assessment": dict matching NoveltyAssessment schema, or None
        "assessment_unavailable_reason": str or None

    Reuses the same LLM pipeline, caching, and grounding logic as
    get_node_details() but skips map-specific steps (connected works,
    timeline, access info, summary/keywords).
    """
    with engine.connect() as conn:
        # Load work metadata
        work_data = load_work_data(conn, work_id)
        if not work_data:
            raise ValueError("work_not_found")

        # Enrich abstract if invalid
        enrichment_happened = False
        enriched_abstract, abstract_source = ensure_valid_abstract(
            conn,
            work_id=work_id,
            title=work_data["title"],
            abstract=work_data["abstract"],
            year=work_data["year"],
            doi=work_data.get("doi"),
            arxiv_id=work_data.get("arxiv_id"),
        )
        if abstract_source not in ("cached", "unavailable") and enriched_abstract:
            if enriched_abstract != work_data["abstract"]:
                logger.info(f"Enriched abstract for {work_id} from {abstract_source}")
                work_data["abstract"] = enriched_abstract
                enrichment_happened = True
        elif abstract_source == "unavailable":
            if work_data["abstract"] is not None:
                logger.info(f"Abstract for {work_id} is invalid and unfetchable, clearing")
                work_data["abstract"] = None
                enrichment_happened = True

        # Infer topic if missing (needed for landmark retrieval)
        inferred_topic_id, topic_source = ensure_topic(
            conn,
            work_id=work_id,
            title=work_data["title"],
            abstract=work_data["abstract"],
            current_topic_id=work_data.get("primary_topic_id"),
            doi=work_data.get("doi"),
            arxiv_id=work_data.get("arxiv_id"),
        )
        if topic_source not in ("cached", "unavailable") and inferred_topic_id:
            if inferred_topic_id != work_data.get("primary_topic_id"):
                logger.info(f"Inferred topic for {work_id}: {inferred_topic_id} from {topic_source}")
                work_data["primary_topic_id"] = inferred_topic_id
                enrichment_happened = True

        # Check cache (skip if enrichment happened — regenerate)
        if not enrichment_happened:
            cached = get_cached_details(conn, work_id)
            if cached and cached.get("novelty_assessment"):
                logger.info(f"Cache hit for novelty assessment: {work_id}")
                return {
                    "novelty_assessment": cached["novelty_assessment"],
                    "assessment_unavailable_reason": None,
                }
            if cached and cached.get("assessment_unavailable_reason"):
                return {
                    "novelty_assessment": None,
                    "assessment_unavailable_reason": cached["assessment_unavailable_reason"],
                }

        # Fetch grounding papers
        referenced_works = get_referenced_works(conn, work_id)
        landmarks = get_topic_landmarks(
            conn,
            work_data.get("primary_topic_id"),
            work_data.get("year"),
        )

        # Detect pioneering work
        cited_by_count = work_data.get("cited_by_count", 0)
        work_year = work_data.get("year")
        is_pioneering = False
        if work_year is not None and cited_by_count > 0:
            paper_age = max(2025 - work_year, 1)
            cites_per_year = cited_by_count / paper_age
            if cited_by_count > 50000:
                is_pioneering = True
            elif work_year < 2000 and cited_by_count > 5000 and len(referenced_works) < 10:
                is_pioneering = True
            elif cites_per_year > 500 and len(referenced_works) < 10:
                is_pioneering = True

        # Handle zero grounding data — try LLM landmark fallback
        if len(referenced_works) == 0 and len(landmarks) == 0:
            if cited_by_count > 500:
                _, fallback_landmarks = supplement_grounding_papers(
                    title=work_data["title"],
                    abstract=work_data["abstract"],
                    year=work_data["year"],
                    existing_refs=[],
                    existing_landmarks=[],
                    field=work_data.get("category"),
                    primary_topic_id=work_data.get("primary_topic_id"),
                    is_pioneering=is_pioneering,
                    target_work_id=work_id,
                    conn=conn,
                )
            else:
                fallback_landmarks = []

            if fallback_landmarks:
                landmarks = fallback_landmarks
            else:
                unavailable_reason = (
                    "Insufficient reference data - no citations or field landmark papers "
                    "available for comparison. Novelty assessment requires at least one "
                    "reference or landmark paper to ground the analysis."
                )
                cache_details(conn, work_id, "", [], None, assessment_unavailable_reason=unavailable_reason)
                return {
                    "novelty_assessment": None,
                    "assessment_unavailable_reason": unavailable_reason,
                }

        # Cross-domain filtering
        target_topic_id = work_data.get("primary_topic_id")
        target_field_name = get_field_from_topic_id(target_topic_id)
        filtered_refs, filtered_landmarks = _filter_cross_domain_papers(
            referenced_works, landmarks, target_field_name, target_topic_id,
            target_title=work_data.get("title"),
            target_abstract=work_data.get("abstract"),
        )

        # Supplement grounding if insufficient
        additional_refs, additional_landmarks = supplement_grounding_papers(
            title=work_data["title"],
            abstract=work_data["abstract"],
            year=work_data["year"],
            existing_refs=filtered_refs,
            existing_landmarks=filtered_landmarks,
            field=work_data.get("category"),
            primary_topic_id=work_data.get("primary_topic_id"),
            is_pioneering=is_pioneering,
            target_work_id=work_id,
            conn=conn,
        )

        if additional_refs or additional_landmarks:
            combined_refs = filtered_refs + additional_refs
            combined_landmarks = filtered_landmarks + additional_landmarks
            filtered_refs, filtered_landmarks = _filter_cross_domain_papers(
                combined_refs, combined_landmarks, target_field_name, target_topic_id,
                target_title=work_data.get("title"),
                target_abstract=work_data.get("abstract"),
            )

        # Attempt to fetch full text for richer novelty assessment
        full_text_sections = None
        context_depth = "abstract_only"

        try:
            cached_ft = conn.execute(
                text("SELECT methods_text, full_text_available FROM paper_full_text_cache WHERE work_id = :wid"),
                {"wid": work_id},
            ).mappings().first()

            if cached_ft and cached_ft["full_text_available"] and cached_ft["methods_text"]:
                full_text_sections = extract_paper_sections(cached_ft["methods_text"])
                context_depth = "full_text"
                logger.info(f"Using cached full text for {work_id}")
            else:
                # Try to download PDF
                pdf_url = work_data.get("oa_pdf_url")
                if not pdf_url and work_data.get("arxiv_id"):
                    pdf_url = f"https://arxiv.org/pdf/{work_data['arxiv_id']}.pdf"

                if pdf_url:
                    logger.info(f"Attempting PDF download for novelty assessment: {pdf_url[:80]}")
                    full_text = download_and_extract_pdf(pdf_url)
                    if full_text:
                        full_text_sections = extract_paper_sections(full_text)
                        context_depth = "full_text"
                        logger.info(f"Full text extracted for {work_id}: {sum(len(v) for v in full_text_sections.values())} chars")
                        # Cache for future use
                        try:
                            conn.execute(
                                text("""
                                    INSERT INTO paper_full_text_cache (work_id, methods_text, full_text_available, fetched_at)
                                    VALUES (:wid, :text, true, now())
                                    ON CONFLICT (work_id) DO UPDATE SET
                                        methods_text = EXCLUDED.methods_text,
                                        full_text_available = true,
                                        fetched_at = now()
                                """),
                                {"wid": work_id, "text": full_text[:50000]},
                            )
                            conn.commit()
                        except Exception as e:
                            logger.warning(f"Failed to cache full text for {work_id}: {e}")
        except Exception as e:
            logger.warning(f"Full text retrieval failed for {work_id}: {e}")

        # LLM assessment
        llm_result = generate_node_details_llm(
            title=work_data["title"] or "Untitled",
            abstract=work_data["abstract"],
            year=work_data["year"],
            referenced_works=filtered_refs,
            landmarks=filtered_landmarks,
            target_work_id=work_id,
            cited_by_count=cited_by_count,
            full_text_sections=full_text_sections,
        )

        # Cache
        cache_details(
            conn, work_id,
            llm_result["summary"],
            llm_result["keywords"],
            llm_result["novelty_assessment"],
        )

        return {
            "novelty_assessment": llm_result["novelty_assessment"],
            "assessment_unavailable_reason": None,
            "context_depth": context_depth,
        }
