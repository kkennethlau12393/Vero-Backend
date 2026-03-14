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
from app.feature3.json_utils import extract_json_from_llm_response
from app.feature3.paper_identity import title_word_overlap, content_word_overlap
from app.feature3.landmark_retrieval import get_topic_landmarks
from app.feature3.node_timeline import build_node_timeline
from app.feature3.novelty_validation import validate_novelty_level
from app.feature3.reference_store import get_referenced_works
from app.feature3.schemas import (
    ConnectedWork,
    EraCommentary,
    GroundingPaper,
    NodeDetailsResponse,
    NodeTimeline,
    NoveltyAssessment,
    ResearchLineageNarrative,
    TimelinePaper,
    TimelineSection,
)
from app.feature3.topic_inference import ensure_topic
from app.feature3.topic_lookup import get_topic_display_name
from app.shared.pdf_utils import download_and_extract_pdf

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Bump this when model OR prompt changes to auto-invalidate cached assessments
ASSESSMENT_VERSION = "maverick-v17"


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


# ---------------------------------------------------------------------------
# Permanent novelty assessment storage (version-gated by ASSESSMENT_VERSION)
# ---------------------------------------------------------------------------

def get_persisted_assessment(
    conn: Connection, work_id: str
) -> Optional[Dict[str, Any]]:
    """Read a permanently stored novelty assessment (version-gated)."""
    try:
        row = conn.execute(
            text("""
                SELECT novelty_level, confidence, whats_new,
                       compared_to_prior_work, novelty_explanation,
                       grounding_papers, context_depth,
                       assessment_unavailable_reason
                FROM novelty_assessments
                WHERE work_id = :work_id
                  AND model_version = :model_version
            """),
            {"work_id": work_id, "model_version": ASSESSMENT_VERSION},
        ).mappings().first()

        if not row:
            return None

        if row["assessment_unavailable_reason"]:
            return {
                "novelty_assessment": None,
                "assessment_unavailable_reason": row["assessment_unavailable_reason"],
            }

        return {
            "novelty_assessment": {
                "novelty_level": row["novelty_level"],
                "confidence": row["confidence"],
                "whats_new": row["whats_new"],
                "compared_to_prior_work": row["compared_to_prior_work"],
                "novelty_explanation": row["novelty_explanation"],
                "grounding_papers": row["grounding_papers"] or [],
                "context_depth": row["context_depth"] or "abstract_only",
            },
            "assessment_unavailable_reason": None,
        }
    except Exception as e:
        logger.warning(f"Failed to read persisted assessment for {work_id}: {e}")
        return None


def persist_assessment(
    conn: Connection,
    work_id: str,
    novelty_assessment: Optional[Dict[str, Any]],
    assessment_unavailable_reason: Optional[str] = None,
) -> None:
    """Permanently store a novelty assessment."""
    try:
        if novelty_assessment:
            conn.execute(
                text("""
                    INSERT INTO novelty_assessments
                        (work_id, novelty_level, confidence, whats_new,
                         compared_to_prior_work, novelty_explanation,
                         grounding_papers, context_depth,
                         model_version, assessment_unavailable_reason)
                    VALUES
                        (:work_id, :novelty_level, :confidence, :whats_new,
                         :compared_to_prior_work, :novelty_explanation,
                         :grounding_papers, :context_depth,
                         :model_version, NULL)
                    ON CONFLICT (work_id) DO UPDATE SET
                        novelty_level = EXCLUDED.novelty_level,
                        confidence = EXCLUDED.confidence,
                        whats_new = EXCLUDED.whats_new,
                        compared_to_prior_work = EXCLUDED.compared_to_prior_work,
                        novelty_explanation = EXCLUDED.novelty_explanation,
                        grounding_papers = EXCLUDED.grounding_papers,
                        context_depth = EXCLUDED.context_depth,
                        model_version = EXCLUDED.model_version,
                        assessment_unavailable_reason = NULL,
                        updated_at = now()
                """),
                {
                    "work_id": work_id,
                    "novelty_level": novelty_assessment.get("novelty_level", "medium"),
                    "confidence": novelty_assessment.get("confidence", "medium"),
                    "whats_new": novelty_assessment.get("whats_new"),
                    "compared_to_prior_work": novelty_assessment.get("compared_to_prior_work"),
                    "novelty_explanation": novelty_assessment.get("novelty_explanation", ""),
                    "grounding_papers": json.dumps(novelty_assessment.get("grounding_papers", [])),
                    "context_depth": novelty_assessment.get("context_depth", "abstract_only"),
                    "model_version": ASSESSMENT_VERSION,
                },
            )
        else:
            conn.execute(
                text("""
                    INSERT INTO novelty_assessments
                        (work_id, novelty_level, confidence, novelty_explanation,
                         model_version, assessment_unavailable_reason)
                    VALUES
                        (:work_id, 'medium', 'low', '',
                         :model_version, :reason)
                    ON CONFLICT (work_id) DO UPDATE SET
                        assessment_unavailable_reason = EXCLUDED.assessment_unavailable_reason,
                        model_version = EXCLUDED.model_version,
                        updated_at = now()
                """),
                {
                    "work_id": work_id,
                    "model_version": ASSESSMENT_VERSION,
                    "reason": assessment_unavailable_reason or "assessment_failed",
                },
            )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to persist assessment for {work_id}: {e}")


def delete_persisted_assessment(conn: Connection, work_id: str) -> None:
    """Delete a persisted assessment (used by force_regenerate)."""
    try:
        conn.execute(
            text("DELETE FROM novelty_assessments WHERE work_id = :work_id"),
            {"work_id": work_id},
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to delete persisted assessment for {work_id}: {e}")


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


def _fetch_external_work_data(work_id: str) -> Optional[Dict[str, Any]]:
    """Fetch work metadata from S2/ArXiv APIs for non-OpenAlex IDs."""
    from app.feature1.citation_map_service import fetch_seed_paper_details

    data = fetch_seed_paper_details(work_id)
    if not data:
        return None

    return {
        "work_id": data.get("work_id", work_id),
        "title": data.get("title"),
        "year": data.get("year"),
        "cited_by_count": int(data.get("cited_by_count") or 0),
        "authors": data.get("authors") or [],
        "venue": data.get("venue"),
        "abstract": data.get("abstract"),
        "primary_topic_id": None,
        "category": None,
        "category_confidence": None,
        "doi": data.get("doi"),
        "arxiv_id": None,
        "is_open_access": data.get("is_open_access"),
        "oa_status": None,
        "oa_pdf_url": None,
    }


def load_work_data(conn: Connection, work_id: str) -> Optional[Dict[str, Any]]:
    """Load work metadata from the works table.

    Falls back to S2/ArXiv APIs for non-OpenAlex IDs (S2:, AX: prefixes).
    """
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
        # Fallback: fetch from external API for non-OpenAlex IDs
        if work_id.startswith("S2:") or work_id.startswith("AX:"):
            return _fetch_external_work_data(work_id)
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
    full_text_sections: Optional[str] = None,
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
        full_text_block = "\n\nFULL TEXT (extracted from PDF — use this for detailed technical analysis):\n" + full_text_sections

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

## GLOBAL CITATION RULE (applies to ALL text fields: whats_new, compared_to_prior_work, novelty_explanation)
You may ONLY cite papers from the TWO lists above: "Papers This Work Cites (References)" AND "Landmark Papers in This Field". Both lists are equally citable. ALWAYS use their work_id format: [W2163605009].
Landmark papers are NOT just background context — they are first-class citable papers. You MUST cite landmark papers in your text fields, not just references.
NEVER use in-paper reference numbers like [3], [28], [22] — the reader cannot look those up.
NEVER mention papers by name alone (e.g., "ManiReg", "DeepWalk") without a work_id — if a paper is not in the lists above, do NOT cite it at all.
NEVER write a paper's title in the prose — the [work_id] renders as a clickable chip showing the title.
BAD: "Crenshaw's Mapping the Margins [W2163605009] exposed..."
GOOD: "[W2163605009] exposed the analytical blind spot..."

## TECHNICAL DEPTH RULE
Each reference and landmark above includes an abstract. USE these abstracts to write technically specific comparisons.
When describing what a grounding paper did, reference the SPECIFIC method, architecture, or finding described in its abstract — not a vague summary.
BAD: "Prior work used graph-based methods" — GOOD: "GCN [W2163605009] introduced spectral-domain convolutions using a first-order Chebyshev approximation of graph Laplacian filters"

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

**Q2: Is this paper PIONEERING? (Did it create an entirely new research task or field?)**
(Only if Q1 = NO)

This question ONLY determines whether the paper is "pioneering". It does NOT assign "high" or "medium". If the paper is not pioneering, proceed to Q3.

TEST: Do the grounding papers address the SAME TASK as the target paper?

If YES (grounding papers work on the same task/problem):
→ NOT pioneering. Proceed to Q3.

If NO (grounding papers are from fundamentally different domains):
→ Did this paper create a task/application that NO prior paper attempted?
→ "pioneering" ONLY if the APPLICATION itself is new, not just the method
→ Example: First paper on neural artistic style transfer → pioneering (nobody had attempted this task before)

PIONEERING requires ALL of:
1. A task/application that NOBODY was working on before
2. The grounding papers are from different domains being COMBINED into something new
3. NOT just a new method for an existing task
4. NOT a dataset, benchmark, tool, or resource

If pioneering → assign "pioneering" and STOP.
If NOT pioneering → proceed to Q3. Do NOT assign any level here.

**Q3: Assign the novelty level (HIGH vs MEDIUM vs LOW)**
(Only if Q0 = NO, Q1 = NO, and Q2 = not pioneering)

**START: This paper is MEDIUM.** This is the default. Most papers — including excellent, well-cited, influential papers — are medium. Being medium is not a criticism.

Your job: determine whether hard evidence forces you to change the level to LOW or HIGH.

**STEP 1 — Check for LOW first:**
LOW = applies existing methods without significant contribution:
- Applies off-the-shelf methods without modification (ran scikit-learn, used standard CNN, applied BERT out of the box)
- Negative results, failed replications, null findings
- Position papers, commentaries, editorials with no empirical work
- Data collection or annotation described without novel methodology
- Workshop or short papers sketching ideas without full implementation or evaluation
If LOW → assign "low" and STOP.

**STEP 2 — The paper is MEDIUM. This is your answer unless STEP 3 forces a change.**

MEDIUM = builds upon an existing approach with significant improvement. This is the vast majority of research — roughly 70-80% of all papers are MEDIUM:
- New method for an existing task (even if it achieves SOTA)
- Applying an existing technique to a new domain or dataset
- Variant or extension of an existing method (new module, different loss, combining known components)
- Incremental or significant improvements on benchmarks
- Empirical study or comparison of existing approaches
- Framework that organizes or unifies existing methods
- Influential datasets, benchmarks, or resources (even highly cited ones like ImageNet, CIFAR, COCO)
- Any paper where the core contribution is applying METHOD X to DOMAIN Y
- Combining two existing techniques into a new hybrid (e.g., "attention + GNN", "BERT + CRF")
- Adapting an architecture from one domain to another (e.g., transformers for vision, CNNs for graphs)

BEING MEDIUM IS NOT A CRITICISM. Most excellent, highly-cited, field-shaping papers are MEDIUM.

**STEP 3 — Check for HIGH (roughly 10-20% of papers):**

HIGH = introduced a fundamentally new class of method that did not previously exist for this problem.

CALIBRATION: In a batch of 20 random papers across all fields, expect 2-4 to be HIGH. The vast majority (70-80%) should be MEDIUM. When in doubt, the answer is MEDIUM.

HIGH requires ALL FOUR of:
1. The paper introduces a FUNDAMENTALLY NEW method — not a variant, extension, combination, or adaptation of existing methods
2. NO prior paper used this CLASS of approach for this problem — not just "nobody used this exact variant"
3. The grounding papers working on the SAME problem use ENTIRELY DIFFERENT methodological families (e.g., they all use RNNs and this paper introduces attention; they all use spectral methods and this paper introduces spatial convolutions)
4. The method created a new research direction that others subsequently built upon

COMMON TRAPS — These are MEDIUM, not HIGH (read carefully):
- "First to apply [existing method] to [specific domain]" → MEDIUM. Applying transformers to protein folding, BERT to legal text, GANs to medical imaging — these are domain adaptations, not new methods.
- "First to combine [method A] + [method B]" → MEDIUM. Combining attention with GNNs, CRF with BERT, skip connections with dense connections — these are hybrid extensions.
- "First to use [specific variant] of [general technique]" → MEDIUM. A specific attention variant, a new normalization scheme, a different pooling strategy — these are improvements within an existing paradigm.
- "Achieves SOTA on [benchmark]" → MEDIUM. Performance does not determine novelty.
- "Nobody did EXACTLY this before" → Almost always MEDIUM. You can make any paper "first" by defining the method+problem narrowly enough. HIGH requires that the METHOD CLASS is new for this problem, not just the specific variant.

SELF-CHECK before assigning HIGH: Ask yourself — is this paper more like ResNet (introduced an entirely new architectural principle that no prior image classification network used) or more like DenseNet (clever extension of ResNet's skip connections)? ResNet = HIGH. DenseNet = MEDIUM. Most papers are DenseNet-like, not ResNet-like.

KEY DISTINCTION — HIGH vs MEDIUM:
- "Attention Is All You Need" → HIGH (eliminated recurrence entirely — a fundamentally new architecture class)
- "BERT" → MEDIUM (applied the existing transformer architecture with masked language modeling — clever, influential, but builds on [the transformer])
- "ResNet" → HIGH (skip connections were a fundamentally new architectural principle for deep networks)
- "DenseNet" → MEDIUM (extends skip connections with dense connectivity — builds on ResNet)
- "AlphaFold 2" → HIGH (introduced a fundamentally new structure prediction approach using attention on MSAs — nothing like prior homology modeling or physics-based methods)
- "ESMFold" → MEDIUM (applies the same attention-based approach as AlphaFold but with a language model — builds on AlphaFold's paradigm)

IMPORTANT: You are classifying the paper's novelty AT THE TIME OF PUBLICATION. But even at time of publication, most papers build on existing approaches.

CITATION COUNT IS NOT A NOVELTY INDICATOR:
- High citations mean IMPACT, not NOVELTY
- BERT has 100K+ citations and is MEDIUM
- ImageNet has 50K+ citations and is MEDIUM
- A highly-cited dataset or benchmark is MEDIUM

**CRITICAL LANGUAGE RULES (READ FIRST):**
BANNED VERBS - NEVER use these in summary, whats_new, explanation, or relevance:
- explores, discusses, examines, investigates, assesses, evaluates, addresses, looks at, studies, analyzes, reviews
- If title uses "Assessing X" → you write "introduces/develops a method for X"
- If title uses "Investigating Y" → you write "establishes/demonstrates Y"

## DETAILED FIELD INSTRUCTIONS — READ CAREFULLY BEFORE WRITING

**WHATS_NEW — What THIS paper introduces (4-6 sentences, 120-200 words)**

SCOPE: Technical mechanism and methodology ONLY. Do NOT describe what the paper enabled for the field or what came after — that belongs in the timeline's contribution_statement.

This field describes ONLY the novel contributions of the target paper. Do NOT mention prior work here — that belongs in compared_to_prior_work.

REQUIRED CONTENT:
- The specific method, framework, architecture, or finding the paper introduces
- Technical details: what mechanism, algorithm, or approach is new
- Name specific components (layers, modules, loss functions, architectures) and explain HOW they work mechanistically — not just what they achieve
- Include mathematical intuition where relevant (e.g., "reformulates layers to learn residual functions F(x) = H(x) - x" rather than "uses residual connections")
- Key results: quantitative improvements, benchmarks achieved, or empirical findings
- Why it matters: what problem does this solve or what limitation does it overcome

Cite 1-2 grounding paper work_ids ONLY to clarify what the new contribution replaces or extends (e.g. "replacing the fully-connected layers used in [W2163605009]"), NOT to describe what those papers did.

ONLY null if novelty_level is "pioneering". NEVER null for reviews — describe what the review SYNTHESIZES or ORGANIZES and what organizational framework it provides.

GOOD "review/survey" whats_new (specific organizational contribution, not just "comprehensive overview"):
"Introduces a new taxonomy for graph neural networks organized along three axes: spectral vs. spatial approaches, the type of graph (homogeneous, heterogeneous, dynamic), and the downstream task (node classification, link prediction, graph classification). Identifies five distinct GNN propagation mechanisms — convolutional [W2163605009], attentional [W2952867780], message-passing [W2614828516], gated recurrence, and sampling-based — and systematically compares their computational complexity, expressiveness, and scalability tradeoffs. Synthesizes benchmark results across 12 standard datasets, revealing that spatial methods consistently outperform spectral methods on heterogeneous graphs while spectral approaches retain advantages on regular lattice structures."

BAD (too shallow):
"Introduces deep residual networks that simplify training of deeper networks, achieving state-of-the-art on ImageNet."

GOOD (specific mechanism, quantitative results, technical depth):
"Introduces residual learning via skip connections that add identity mappings between layers, directly addressing the degradation problem where deeper networks paradoxically produce higher training error. The core innovation reformulates layers to learn residual functions F(x) = H(x) - x rather than the underlying mapping H(x), which is easier to optimize. The residual block consists of two 3x3 convolution layers with batch normalization, where the input is added directly to the output via a shortcut connection that requires no additional parameters. For dimension mismatches, 1x1 convolutions with stride 2 perform linear projection. This enables training networks with 152 layers (8x deeper than VGG [W2109255472]), achieving 3.57% top-5 error and winning ILSVRC 2015. The residual blocks are modular and demonstrated on both classification (ImageNet) and detection (COCO) tasks."

**COMPARED_TO_PRIOR_WORK — How this paper differs from what came before (4-7 sentences, 150-250 words)**

SCOPE: Technical differences between methods ONLY. Do NOT tell the chronological story of field evolution — that belongs in the timeline's historical_context.

This field describes ONLY how the target paper's approach differs from specific prior methods. Do NOT re-describe what the target paper introduces — that belongs in whats_new.

CITATION RULES (CRITICAL — READ BEFORE WRITING):
- You may ONLY cite papers from the reference list AND the landmark list provided above — BOTH are equally citable
- ALWAYS cite by work_id: [W2163605009], [W2109255472], etc.
- NEVER cite by in-paper reference number: [3], [28], [22] — the reader has NO access to the paper's bibliography
- NEVER cite papers by name only without a work_id: "ManiReg", "DeepWalk", "SemiEmb" — if a paper is not in the reference or landmark lists, do NOT cite it
- Cite at least 3 UNIQUE papers (3 DIFFERENT work_ids) — must include at least 1 from the LANDMARK list
- SPREAD citations across BOTH references AND landmarks — do NOT only cite the first 1-2 references while ignoring landmarks. Landmarks provide essential field context.

REQUIRED CONTENT:
- For each grounding paper (reference OR landmark) that is technically relevant, dedicate a sentence explaining what THAT paper specifically did (its approach, its limitation). You have 5-7 grounding papers — use them.
- For each cited prior paper, name its SPECIFIC technical approach (e.g., "used max-pooling over per-point features for global representation" not just "processed point clouds")
- Describe the specific technical limitation (e.g., "could not capture local geometric relationships between neighboring points" not just "had limitations")
- Explain the concrete technical difference between prior approaches and this paper
- If applicable, quantify the improvement (accuracy gains, speed improvements, capability gaps filled)

Do NOT group papers — each paper gets its own explanation.

Do NOT cite a paper just to fill space. Every citation must provide unique technical context that helps the reader understand what existed before and why this paper's approach is different.

BAD (groups papers, no specific detail):
"Prior work such as PointNet [W1] and PointNet++ [W2] processed point clouds."

GOOD (each paper gets its own specific explanation):
"PointNet [W1] introduced per-point MLPs with max-pooling to extract global features from unordered point sets, but could not capture local geometric structure. PointNet++ [W2] addressed this by adding hierarchical grouping with ball queries at multiple scales, though the fixed radius grouping struggled with varying point densities. DGCNN [W3] replaced the fixed radius grouping with dynamic edge convolutions that recompute nearest neighbors at each layer, but required O(n*k) memory for k-nearest neighbor graphs."

ONLY null if novelty_level is "pioneering". For reviews: explain what SPECIFIC aspects of each prior survey's coverage, scope, or taxonomy this review extends, corrects, or re-organizes.

GOOD "review/survey" compared_to_prior_work (specific coverage gaps and organizational differences):
"The earlier GNN survey by Wu et al. [W2912389459] organized methods into four categories (recurrent, convolutional, graph autoencoders, spatial-temporal) but did not cover attention-based propagation mechanisms like GAT [W2952867780] or the emerging sampling-based approaches for large-scale graphs. Zhou et al. [W2899632714] provided broader coverage of GNN variants but focused on architectural descriptions without systematic computational complexity analysis. Bronstein et al. [W2614828516] established the geometric deep learning framework connecting CNNs, GNNs, and transformers under a common mathematical lens, but their treatment predated key developments in dynamic graph networks and heterogeneous graph learning. This survey addresses these gaps by introducing a three-axis taxonomy (spectral/spatial, graph type, task type) that accommodates all five propagation mechanisms and includes complexity benchmarks missing from prior surveys."

**NOVELTY_EXPLANATION — Summary justification of the novelty classification (4-6 sentences, 120-200 words)**

SCOPE: Classification justification ONLY. Do NOT describe downstream impact or what successors built — that belongs in the timeline.

This field synthesizes findings from whats_new and compared_to_prior_work to justify WHY the assigned novelty level is correct. It should read as a self-contained justification.

CITATION RULES (same as compared_to_prior_work):
- You may ONLY cite papers from the reference list AND the landmark list — BOTH are equally citable
- ALWAYS cite by work_id: [W2163605009], [W2109255472], etc.
- NEVER cite by in-paper reference number: [3], [28], [22] — the reader has NO access to the paper's bibliography
- NEVER cite papers by name only without a work_id — if a paper is not in the reference or landmark lists, do NOT cite it
- Cite at least 3 UNIQUE papers — prefer citing DIFFERENT papers than those in compared_to_prior_work, including LANDMARKS
- SPREAD citations across both references AND landmarks

REQUIRED CONTENT:
- State the novelty level and primary reason in the first sentence
- Cite papers from BOTH the reference and landmark lists where they strengthen the argument. Reference DIFFERENT papers than those emphasized in compared_to_prior_work to demonstrate breadth of evidence
- Each citation must explain what that paper established and how it relates to the novelty classification argument. Do NOT cite papers that don't add to the justification
- Explain the causal chain: what existed before (with citations) → what this paper changed → why that warrants this level
- For "pioneering": explain what task/field DID NOT EXIST before this paper, cite grounding papers from DIFFERENT domains that were combined, and explain why no prior paper attempted this specific application
- For "high": explain why not "pioneering" (task existed before) and not "medium" (this paper was the FIRST to apply this method to this problem — no prior paper did it)
- For "medium": explain why the contribution builds on existing approaches rather than being first-of-its-kind — what existing method/framework was applied, and what prior paper already did something similar
- For "low": explain why the contribution lacks methodological novelty — what off-the-shelf method was used without modification

FORBIDDEN patterns (will fail validation):
- "Classified as X due to title containing..."
- "Classified as review due to its synthesis..."
- Generic: "builds upon prior work and updates the field"
- Generic: "provides a comprehensive overview" or "valuable resource for researchers"
- Vague comparison: "builds upon existing surveys such as X and Y" without explaining WHAT it adds

BAD (circular reasoning, no technical substance):
"Classified as high because it introduces deep residual learning that builds upon previous architectures [W2163605009] and updates traditional approaches."

BAD (review — vague, no specifics about what the survey actually organizes or how it extends prior work):
"Classified as medium because this survey provides a comprehensive overview of GNNs, synthesizing existing knowledge. While it does not introduce a new method, it provides a valuable resource. The survey builds upon existing surveys [W2912389459] and [W2899632714]."

GOOD "high" (clear causal chain, specific claims, cited evidence):
"Classified as high because while deep image classification networks already existed — AlexNet [W2163605009] demonstrated 8-layer CNNs and VGGNet [W2109255472] showed depth improves accuracy up to 19 layers — NO prior paper solved training of very deep networks (100+ layers) via identity shortcut connections. Highway Networks [W2153625789] attempted gated information flow but required learned gating parameters and could not scale beyond ~50 layers. ResNet's parameter-free skip connections were the first method to successfully train 152-layer networks, reducing top-5 error from 7.3% to 3.57%. This is not pioneering because image classification with deep CNNs was well-established by 2015."

GOOD "medium" (acknowledges solid contribution while explaining why it's not high):
"Classified as medium because this paper applies the established transformer architecture [W2118176668] to medical image segmentation, adapting the self-attention mechanism with domain-specific preprocessing for CT scans. Vision transformers for dense prediction were already demonstrated by ViT [W3035667763] for image classification and Swin Transformer [W3134447684] for hierarchical vision tasks. TransUNet [W3128763281] had already applied transformers to medical image segmentation before this paper. The contribution is a domain-specific refinement of existing transformer-based segmentation approaches."

GOOD "medium (review/survey)" (explains WHAT the review organizes and HOW it extends prior surveys):
"Classified as medium because this is a survey (Q0) that synthesizes existing GNN methods without introducing new architectures or algorithms. However, it provides substantial organizational value: its three-axis taxonomy (spectral/spatial, graph type, task) extends the four-category framework of Wu et al. [W2912389459] by adding attention-based and sampling-based propagation as distinct categories. It also fills gaps in Zhou et al. [W2899632714] by including systematic computational complexity analysis across all five propagation mechanisms. The survey covers 150+ papers spanning 2013-2023 and identifies that spatial methods [W2163605009] outperform spectral approaches on heterogeneous graphs — a finding not established in prior surveys. Classified as medium rather than low because the new taxonomy and cross-method benchmarking represent a meaningful organizational contribution beyond simple compilation."

GOOD "pioneering" (demonstrates a task/field that did not exist before):
"Classified as pioneering because no prior work attempted neural artistic style transfer — the task of rendering a photograph in the style of a painting while preserving content. Gatys et al. combined convolutional feature representations from VGGNet [W2109255472], originally developed for object classification, with a Gram-matrix-based style representation that had no precedent in computer vision. Prior texture synthesis methods [W2100339939] operated on low-level statistics without separating content from style. This paper created an entirely new research direction that combined two previously unrelated domains."

GOOD "low" (explains lack of methodological novelty):
"Classified as low because this paper applies standard logistic regression and random forest classifiers [W2034096913] to a customer churn dataset without modification to the algorithms or training procedure. The feature engineering follows established practices from prior work [W2056891283], and the evaluation uses standard accuracy/AUC metrics on a single proprietary dataset. No new method, insight, or benchmark is introduced."

---

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
        "whats_new": "Follow WHATS_NEW instructions above. 4-6 sentences, 120-200 words. Mechanistic depth required.",
        "compared_to_prior_work": "Follow COMPARED_TO_PRIOR_WORK instructions above. 4-7 sentences, 150-250 words. Each cited paper gets its own explanation.",
        "novelty_explanation": "Follow NOVELTY_EXPLANATION instructions above. 4-6 sentences, 120-200 words. Must explain causal chain with citations.",
        "grounding_papers": [
            {{
                "work_id": "W...",
                "title": "...",
                "year": 2020,
                "cited_by_count": 1000,
                "relationship": "cited_reference" | "field_landmark",
                "relevance": "NEVER use banned verbs. MUST explain the specific technical contribution of THIS paper and how the target paper relates to it. BAD: 'Landmark paper in deep learning'. GOOD: 'Introduced 8-layer CNN architecture for ImageNet classification, establishing that deep networks outperform hand-crafted features — the target paper extends this by enabling 152-layer training via skip connections'."
            }}
        ]
    }}
}}

**GROUNDING PAPERS — SELECTION AND QUALITY RULES:**
- Include 5-7 papers (mix of cited_reference + field_landmark)
- ONLY use work_ids from the reference and landmark lists provided above
- ONLY include papers that are DIRECTLY technically related to the target paper's method, architecture, or findings
- Do NOT include papers just because they are highly cited — they must be relevant to the target paper's specific contribution
- If a reference or landmark is from a different sub-field or uses unrelated methods, EXCLUDE it from grounding_papers
- Each relevance MUST explain what the grounding paper technically contributed AND how the target paper relates to it
- BAD: "Landmark paper in this field" or "Important work in machine learning"
- GOOD: "Established SNe Ia as standard candles for cosmic distance measurement, which this paper directly uses to constrain the dark energy equation of state"
- Prefer papers the target paper METHODOLOGICALLY builds upon or EMPIRICALLY compares against

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
- ONLY cite grounding paper work_ids (e.g., [W2163605009]) from the lists above — NEVER use in-paper reference numbers [3], [28]. NEVER truncate work_ids.
- Aim for 3+ UNIQUE work_id citations (3 different grounding papers, not repeats)
- MUST make specific technical claims (what method? what finding? what improvement?)
- FORBIDDEN patterns (will fail validation):
  - "Classified as X due to title containing..."
  - "Classified as review due to its synthesis..."
  - "The paper is X because it doesn't create..."
  - Citing papers by in-paper reference numbers: "[3]", "[28]", "[22]"
  - Citing papers by name without work_id: "ManiReg", "DeepWalk", "SemiEmb"
- REQUIRED pattern: "Classified as X because [specific technical evidence with work_id citations]"

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
    full_text_sections: Optional[str] = None,
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
                        "ONLY cite papers from the provided reference/landmark lists using their FULL work_id (e.g., [W2163605009]). NEVER truncate or shorten work_ids. "
                        "NEVER use in-paper reference numbers like [3] or [28]. "
                        "NEVER cite papers not in the provided lists.",
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
    """Filter cross-domain papers from landmarks only.

    References are NEVER filtered — the author cited them, so they are
    inherently relevant regardless of OpenAlex field classification.
    OpenAlex topic assignments can be wrong (e.g., a robotics paper
    classified under "Automotive Engineering"), and filtering references
    based on potentially-wrong topics removes correct papers.

    Landmarks are filtered using content overlap with the target paper's
    title and abstract, NOT by subfield comparison. This ensures that
    even when the target paper's topic_id is wrong, relevant landmarks
    survive and irrelevant ones are removed.

    Args:
        referenced_works: Papers the target work cites (never filtered)
        landmarks: Field landmark papers (filtered by content relevance)
        target_field_name: OpenAlex subfield name (unused, kept for API compat)
        target_topic_id: OpenAlex topic ID (unused, kept for API compat)
        target_title: Title of target paper for content overlap check
        target_abstract: Abstract of target paper for content overlap check

    Returns:
        (referenced_works, filtered_landmarks)
    """
    # References are author-curated — never filter them
    # The author chose to cite these papers, so they're relevant by definition
    filtered_refs = referenced_works

    # Filter landmarks using content overlap with the target paper
    # This is robust to wrong topic_ids because it checks actual content
    target_text = f"{target_title or ''} {target_abstract or ''}"
    if not target_text.strip():
        return filtered_refs, landmarks

    filtered_landmarks = []
    for lm in landmarks:
        lm_title = lm.get("title", "")
        lm_abstract = lm.get("abstract", "")
        lm_text = f"{lm_title} {lm_abstract}"

        # Check content overlap between landmark and target paper
        # Uses title-to-title overlap (most reliable signal)
        title_overlap = content_word_overlap(target_title or "", lm_title) if target_title and lm_title else 0

        # Also check abstract overlap if available (catches cases where
        # titles are different but papers are about the same topic)
        abstract_overlap = 0
        if target_abstract and lm_abstract:
            abstract_overlap = content_word_overlap(target_abstract, lm_abstract)

        # Keep landmark if it has meaningful content overlap with target
        # Title overlap >= 0.05 OR abstract overlap >= 0.05
        if title_overlap >= 0.05 or abstract_overlap >= 0.05:
            filtered_landmarks.append(lm)
        else:
            logger.info(
                f"Filtering irrelevant landmark: '{lm_title[:50]}...' "
                f"(title_overlap={title_overlap:.3f}, abstract_overlap={abstract_overlap:.3f})"
            )

    lm_filtered = len(landmarks) - len(filtered_landmarks)
    if lm_filtered > 0:
        logger.info(
            f"Content-based landmark filtering: {len(landmarks)}->{len(filtered_landmarks)} "
            f"(-{lm_filtered} irrelevant)"
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

    # Build authors lookup from refs + landmarks for propagation to grounding papers
    _papers_by_id = {r["work_id"]: r for r in filtered_refs}
    _papers_by_id.update({lm["work_id"]: lm for lm in filtered_landmarks})

    for gp in raw_grounding:
        if isinstance(gp, dict) and gp.get("work_id") in valid_work_ids:
            grounding_papers.append({
                "work_id": gp["work_id"],
                "title": gp.get("title", ""),
                "year": gp.get("year"),
                "cited_by_count": gp.get("cited_by_count"),
                "relationship": gp.get("relationship", "cited_reference"),
                "relevance": gp.get("relevance", ""),
                "authors": _papers_by_id.get(gp["work_id"], {}).get("authors", []),
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
                    "authors": lm.get("authors", []),
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
                    "authors": ref.get("authors", []),
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
                    "authors": lm.get("authors", []),
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
                    "authors": paper.get("authors", []),
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



# _enforce_novelty_level DELETED — replaced by novelty_validation.validate_novelty_level()
# which uses external prior art search instead of citation thresholds.



def _build_narrative_obj(narrative_data: Optional[Dict[str, Any]]) -> Optional[ResearchLineageNarrative]:
    """Convert narrative dict from timeline_narrative.py to Pydantic model."""
    if not narrative_data:
        return None
    return ResearchLineageNarrative(
        historical_context=narrative_data.get("historical_context", ""),
        contribution_statement=narrative_data.get("contribution_statement", ""),
        downstream_impact=narrative_data.get("downstream_impact", ""),
        era_commentaries=[
            EraCommentary(**ec)
            for ec in narrative_data.get("era_commentaries", [])
        ],
        cross_domain_influence=narrative_data.get("cross_domain_influence"),
        paper_type=narrative_data.get("paper_type"),
        is_paradigm_shift=narrative_data.get("is_paradigm_shift", False),
        impact_score=narrative_data.get("impact_score", 0.0),
    )


def _build_timeline_if_requested(
    conn: Connection,
    include_timeline: bool,
    work_id: str,
    work_data: Dict[str, Any],
    referenced_works: Optional[List[Dict[str, Any]]] = None,
    landmarks: Optional[List[Dict[str, Any]]] = None,
) -> Optional[NodeTimeline]:
    """Build timeline object if requested, loading data as needed."""
    if not include_timeline:
        return None

    if referenced_works is None:
        referenced_works = get_referenced_works(conn, work_id)
    if landmarks is None:
        # Check if grounding supplement already cached enriched landmarks
        from app.feature3.paper_cache import get_landmarks as get_cached_landmarks
        cached_landmarks = get_cached_landmarks(conn, work_id)
        if cached_landmarks:
            landmarks = cached_landmarks
        else:
            landmarks = get_topic_landmarks(
                conn, work_data.get("primary_topic_id"),
                work_data.get("year"), target_title=work_data.get("title"),
            )

    timeline_data = build_node_timeline(
        conn, work_id, work_data["year"],
        referenced_works, landmarks,
        include_narrative=True,
        target_title=work_data.get("title"),
        target_abstract=work_data.get("abstract"),
        target_cited_by_count=work_data.get("cited_by_count", 0),
    )

    narrative_data = timeline_data.get("narrative")

    return NodeTimeline(
        target_work_id=timeline_data["target_work_id"],
        target_year=timeline_data["target_year"],
        backward=[
            TimelineSection(
                era=s["era"],
                papers=[TimelinePaper(**p) for p in s["papers"]],
            )
            for s in timeline_data["backward"]
        ],
        forward=[
            TimelineSection(
                era=s["era"],
                papers=[TimelinePaper(**p) for p in s["papers"]],
            )
            for s in timeline_data["forward"]
        ],
        narrative=_build_narrative_obj(narrative_data),
        before_approach=narrative_data.get("before_approach") if narrative_data else None,
        after_approach=narrative_data.get("after_approach") if narrative_data else None,
        shift_description=None,  # Legacy field, narrative has richer content
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
            "authors": top_ref.get("authors", []),
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
            "authors": top_lm.get("authors", []),
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
    map_id: Optional[UUID] = None,
    rank_job_id: Optional[UUID] = None,
    work_id: str,
    include_novelty: bool = False,
    include_timeline: bool = False,
    force_regenerate: bool = False,
) -> NodeDetailsResponse:
    """
    Get detailed pop-up information for a work.

    This is the main entry point for Feature 3. Supports two access paths:
    - map_id: work must be a node in the citation map (returns connected_works)
    - rank_job_id: work must be in the rank job results (no connected_works)

    When include_novelty=False (default), returns lightweight metadata only
    (no LLM call). When True, runs the full novelty assessment pipeline.
    """
    if not map_id and not rank_job_id:
        raise ValueError("Either map_id or rank_job_id is required")

    with engine.connect() as conn:
        if map_id:
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
        else:
            # Verify rank job exists and belongs to tenant
            job_row = conn.execute(
                text("""
                    SELECT rank_job_id FROM rank_jobs
                    WHERE rank_job_id = :rank_job_id AND tenant_id = :tenant_id
                """),
                {"rank_job_id": rank_job_id, "tenant_id": tenant_id},
            ).first()

            if not job_row:
                raise ValueError("rank_job_not_found")

            # Verify work is in this rank job's results
            result_row = conn.execute(
                text("""
                    SELECT work_id FROM rank_results
                    WHERE rank_job_id = :rank_job_id AND work_id = :work_id
                """),
                {"rank_job_id": rank_job_id, "work_id": work_id},
            ).first()

            if not result_row:
                raise ValueError("work_not_in_results")

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

        # Load connected works (from map edges — only available for map-based access)
        connected_works = load_connected_works(conn, map_id, work_id) if map_id else []

        # Abstract is REQUIRED for novelty assessment — LLM produces garbage without it
        if include_novelty and not work_data.get("abstract"):
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
            persist_assessment(
                conn, work_id, None,
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

        # Lightweight metadata-only response (no LLM call for novelty)
        if not include_novelty:
            basic_summary = _generate_summary_from_abstract(
                work_data["abstract"], work_data["title"]
            )
            basic_keywords = _extract_keywords_from_abstract(work_data["abstract"])

            # Build timeline even without novelty
            timeline_obj = _build_timeline_if_requested(
                conn, include_timeline, work_id, work_data,
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
                timeline=timeline_obj,
                primary_topic_id=work_data.get("primary_topic_id"),
                topic_display_name=topic_display_name,
                access_status=access_info["access_status"],
                pdf_url=access_info["pdf_url"],
                doi_url=access_info["doi_url"],
                oa_status=work_data.get("oa_status"),
            )

        # Force regeneration: clear permanent storage so we fall through
        if force_regenerate:
            delete_persisted_assessment(conn, work_id)

        # Check permanent storage first (version-gated by ASSESSMENT_VERSION)
        if not enrichment_happened and not force_regenerate:
            persisted = get_persisted_assessment(conn, work_id)
            if persisted:
                novelty_assessment_obj = None
                if persisted["novelty_assessment"]:
                    novelty_dict = dict(persisted["novelty_assessment"])
                    grounding_papers_data = novelty_dict.pop("grounding_papers", [])
                    novelty_assessment_obj = NoveltyAssessment(
                        **novelty_dict,
                        grounding_papers=[
                            GroundingPaper(**gp)
                            for gp in grounding_papers_data
                        ],
                    )

                # Summary/keywords: try cache, fallback to basic
                cached_for_summary = get_cached_details(conn, work_id)
                if cached_for_summary:
                    summary = cached_for_summary["summary"]
                    keywords = cached_for_summary["keywords"]
                else:
                    summary = _generate_summary_from_abstract(
                        work_data["abstract"], work_data["title"]
                    )
                    keywords = _extract_keywords_from_abstract(work_data["abstract"])

                # Build timeline if requested
                timeline_obj = _build_timeline_if_requested(
                    conn, include_timeline, work_id, work_data,
                )

                return NodeDetailsResponse(
                    work_id=work_data["work_id"],
                    title=work_data["title"],
                    year=work_data["year"],
                    authors=work_data["authors"],
                    venue=work_data["venue"],
                    cited_by_count=work_data["cited_by_count"],
                    abstract=work_data["abstract"],
                    summary=summary,
                    keywords=keywords,
                    novelty_assessment=novelty_assessment_obj,
                    connected_works=connected_works,
                    assessment_unavailable_reason=persisted.get("assessment_unavailable_reason"),
                    timeline=timeline_obj,
                    primary_topic_id=work_data.get("primary_topic_id"),
                    topic_display_name=topic_display_name,
                    access_status=access_info["access_status"],
                    pdf_url=access_info["pdf_url"],
                    doi_url=access_info["doi_url"],
                    oa_status=work_data.get("oa_status"),
                )

        # Check version-gated cache (fallback if not in permanent storage)
        cached = None
        if not enrichment_happened and not force_regenerate:
            cached = get_cached_details(conn, work_id)

        if cached:
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
            timeline_obj = _build_timeline_if_requested(
                conn, include_timeline, work_id, work_data,
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
            target_title=work_data.get("title"),
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
                timeline_obj = _build_timeline_if_requested(
                    conn, include_timeline, work_id, work_data,
                    referenced_works=[], landmarks=[],
                )

                unavailable_reason = (
                    "This paper's reference list is not available in any public academic "
                    "database (often due to publisher restrictions), so we cannot compare "
                    "it against prior work to assess novelty."
                )

                # Cache the unavailable result to avoid repeated expensive lookups
                cache_details(
                    conn, work_id,
                    basic_summary, basic_keywords,
                    None,
                    assessment_unavailable_reason=unavailable_reason,
                )
                persist_assessment(
                    conn, work_id, None,
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
                full_text_sections = cached_ft["methods_text"]
                context_depth = "full_text"
                logger.info(f"Using cached full text for {work_id}: {len(full_text_sections)} chars")
            else:
                # Try to download PDF
                pdf_url = work_data.get("oa_pdf_url")
                if not pdf_url and work_data.get("arxiv_id"):
                    pdf_url = f"https://arxiv.org/pdf/{work_data['arxiv_id']}.pdf"

                if pdf_url:
                    logger.info(f"Attempting PDF download for novelty assessment: {pdf_url[:80]}")
                    full_text = download_and_extract_pdf(pdf_url)
                    if full_text:
                        full_text_sections = full_text
                        context_depth = "full_text"
                        logger.info(f"Full text extracted for {work_id}: {len(full_text)} chars")
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

        # Validate novelty level with external prior art search
        validate_novelty_level(
            llm_result["novelty_assessment"],
            work_data=work_data,
            referenced_works=referenced_works,
            landmarks=landmarks,
        )

        # Cache the result
        cache_details(
            conn,
            work_id,
            llm_result["summary"],
            llm_result["keywords"],
            llm_result["novelty_assessment"],
        )
        # Persist permanently
        persist_assessment(conn, work_id, llm_result["novelty_assessment"])

        # Build timeline if requested
        timeline_obj = _build_timeline_if_requested(
            conn, include_timeline, work_id, work_data,
            referenced_works=referenced_works, landmarks=landmarks,
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


def get_timeline_for_work(
    engine: Engine,
    *,
    work_id: str,
) -> Optional[NodeTimeline]:
    """
    Build timeline for a work_id without requiring a citation map.

    Returns a NodeTimeline Pydantic model, or None if timeline generation fails.
    Reuses the same pipeline as get_node_details(include_timeline=True) but
    skips map-specific steps.
    """
    with engine.connect() as conn:
        work_data = load_work_data(conn, work_id)
        if not work_data:
            raise ValueError("work_not_found")

        # Enrich abstract if invalid (needed for narrative quality)
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
                work_data["abstract"] = enriched_abstract
        elif abstract_source == "unavailable":
            if work_data["abstract"] is not None:
                work_data["abstract"] = None

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
                work_data["primary_topic_id"] = inferred_topic_id

        return _build_timeline_if_requested(
            conn, True, work_id, work_data,
        )


def get_novelty_for_work(
    engine: Engine,
    *,
    work_id: str,
    force_regenerate: bool = False,
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

        # Force regeneration: clear permanent storage
        if force_regenerate:
            delete_persisted_assessment(conn, work_id)

        # Check permanent storage first (no version gate)
        if not enrichment_happened and not force_regenerate:
            persisted = get_persisted_assessment(conn, work_id)
            if persisted:
                logger.info(f"Permanent assessment hit for {work_id}")
                return persisted

        # Check version-gated cache (fallback)
        if not enrichment_happened and not force_regenerate:
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
            target_title=work_data.get("title"),
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
                    "This paper's reference list is not available in any public academic "
                    "database (often due to publisher restrictions), so we cannot compare "
                    "it against prior work to assess novelty."
                )
                cache_details(conn, work_id, "", [], None, assessment_unavailable_reason=unavailable_reason)
                persist_assessment(conn, work_id, None, assessment_unavailable_reason=unavailable_reason)
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
                full_text_sections = cached_ft["methods_text"]
                context_depth = "full_text"
                logger.info(f"Using cached full text for {work_id}: {len(full_text_sections)} chars")
            else:
                # Try to download PDF
                pdf_url = work_data.get("oa_pdf_url")
                if not pdf_url and work_data.get("arxiv_id"):
                    pdf_url = f"https://arxiv.org/pdf/{work_data['arxiv_id']}.pdf"

                if pdf_url:
                    logger.info(f"Attempting PDF download for novelty assessment: {pdf_url[:80]}")
                    full_text = download_and_extract_pdf(pdf_url)
                    if full_text:
                        full_text_sections = full_text
                        context_depth = "full_text"
                        logger.info(f"Full text extracted for {work_id}: {len(full_text)} chars")
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
        # Persist permanently (survives version bumps)
        persist_assessment(conn, work_id, llm_result["novelty_assessment"])

        return {
            "novelty_assessment": llm_result["novelty_assessment"],
            "assessment_unavailable_reason": None,
            "context_depth": context_depth,
        }
