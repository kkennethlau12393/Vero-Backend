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
from app.feature3.grounding_supplement import supplement_grounding_papers
from app.feature3.landmark_retrieval import get_topic_landmarks
from app.feature3.reference_store import get_referenced_works
from app.feature3.schemas import (
    ConnectedWork,
    GroundingPaper,
    NodeDetailsResponse,
    NoveltyAssessment,
)
from app.feature3.topic_inference import ensure_topic

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5
MODEL_VERSION = "gpt-4o-mini"


def get_cached_details(conn: Connection, work_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached node details for a work."""
    try:
        row = conn.execute(
            text("""
                SELECT summary, keywords, novelty_assessment, model_version,
                       assessment_unavailable_reason
                FROM node_details_cache
                WHERE work_id = :work_id
            """),
            {"work_id": work_id},
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
                "model_version": MODEL_VERSION,
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
                   doi, arxiv_id
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
) -> str:
    """Build the LLM prompt with grounded paper context."""
    year_str = f" ({year})" if year else ""
    abstract_text = _truncate_text(abstract or "No abstract available.", 1500)

    # Build references section
    refs_section = ""
    if referenced_works:
        refs_lines = []
        for i, ref in enumerate(referenced_works[:10], 1):
            ref_title = ref.get("title") or "Untitled"
            ref_year = ref.get("year") or "?"
            ref_category = ref.get("category", "")
            category_label = f" [{ref_category}]" if ref_category else ""
            ref_abstract = _truncate_text(ref.get("abstract") or "", 200)
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
            lm_abstract = _truncate_text(lm.get("abstract") or "", 200)
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

## Papers This Work Cites (References)
{refs_section}

## Landmark Papers in This Field (Historical Context)
{landmarks_section}

---

## TECHNICAL DIFFERENTIATION FRAMEWORK

When assessing novelty, identify SPECIFIC technical differences:
1. **METHODOLOGY**: New algorithms, numerical methods, analytical approaches
2. **SCOPE**: Different problem domain, scale, or application area
3. **INTEGRATION**: Combining existing methods in novel ways
4. **VALIDATION**: New experimental/computational validation approaches
5. **THEORETICAL**: New mathematical framework or proofs

For EACH grounding paper you cite, explain:
- What specific technical aspect differs (methodology, scope, theory, etc.)
- Whether this is an incremental or fundamental difference
- How this advances the state of the art

## EXAMPLES OF GOOD SPECIFICITY

✓ GOOD: "Unlike W2525778437 which used recurrent architectures with sequential processing, this work introduces attention-only mechanisms enabling full parallelization"

✓ GOOD: "While W2613904329 applied convolutional sequence learning, this extends to pure attention with multi-head mechanisms"

✗ AVOID: "This work improves upon prior methods" (How? Be specific!)
✗ AVOID: "A novel approach to the problem" (What makes it novel? Cite specific work_ids!)

---

Please provide:
1. A concise summary (2-3 sentences) explaining what this paper does and its main contribution
2. 5-10 keywords/key phrases that capture the paper's main topics
3. A novelty assessment comparing this work to the specific papers listed above

IMPORTANT ASSESSMENT GUIDANCE:
- {landmark_instruction}
- Your assessment MUST reference specific papers by their work_id (e.g., W2525778437)
- Do NOT claim "lack of landmark papers" - if the landmark section shows unavailable, that's a data limitation, not a field characteristic
- Use the technical differentiation framework to be specific about HOW this work differs
- Cite SPECIFIC technical aspects (algorithms, methods, validation approaches)
- Do NOT make claims about prior work without citing specific papers from the provided lists

Return your response as JSON with this exact structure:
{{
    "summary": "...",
    "keywords": ["keyword1", "keyword2", ...],
    "novelty_assessment": {{
        "whats_new": "A short paragraph (3-5 sentences) describing what's new in this paper compared to its references. Explain the key innovations, methodological advances, or novel insights. Cite specific work_ids from the references section.",
        "compared_to_prior_work": "A short paragraph (3-5 sentences) comparing this work to the landmark papers in the field. Explain how it differs from, extends, or challenges prior approaches. Cite specific work_ids from the landmarks section. If no landmarks available, compare to references instead.",
        "novelty_level": "low" | "medium" | "high",
        "confidence": "low" | "medium" | "high",
        "novelty_explanation": "Detailed explanation of novelty (cite specific papers)",
        "grounding_papers": [
            {{
                "work_id": "W...",
                "title": "...",
                "year": 2020,
                "cited_by_count": 1000,
                "relationship": "cited_reference" | "field_landmark",
                "relevance": "How this paper relates to the novelty claim"
            }}
        ]
    }}
}}

Guidelines for novelty_level:
- "low": Incremental improvement or application of existing methods (cite which methods from work_ids)
- "medium": Notable contribution with new insights or methodology (explain what's new vs. work_ids)
- "high": Breakthrough or paradigm-shifting work (explain fundamental departure from work_ids)

Guidelines for grounding_papers:
- Include 5-7 papers that support your novelty assessment (aim for balance: ~3 references + ~3 landmarks)
- Use work_ids from the lists above ONLY
- Include BOTH types: cited_reference (methodology comparison) AND field_landmark (historical context)
- Explain how each paper relates to your assessment (be specific!)"""

    return prompt


def generate_node_details_llm(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    target_work_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call LLM to generate summary, keywords, and grounded novelty assessment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found, returning default response")
        return _default_response(title, referenced_works, landmarks)

    client = OpenAI(api_key=api_key)
    prompt = _build_grounded_prompt(title, abstract, year, referenced_works, landmarks)

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
                temperature=0.3,  # Lower temperature for consistent, focused responses
                timeout=90.0,
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                )

            result = json.loads(content)
            return _validate_llm_response(result, referenced_works, landmarks, target_work_id)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response JSON: {e}")
            if attempt == MAX_RETRIES - 1:
                break
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
    # Extract all work_ids from narrative text
    narrative_text = (
        novelty_assessment.get("whats_new", "") +
        " " + novelty_assessment.get("compared_to_prior_work", "") +
        " " + novelty_assessment.get("novelty_explanation", "")
    )
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


def _validate_llm_response(
    result: Dict[str, Any],
    referenced_works: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    target_work_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and normalize the LLM response."""
    summary = result.get("summary", "Summary unavailable")
    keywords = result.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    novelty = result.get("novelty_assessment", {})

    # Validate grounding papers
    grounding_papers = []
    raw_grounding = novelty.get("grounding_papers", [])
    valid_work_ids = {r["work_id"] for r in referenced_works} | {
        lm["work_id"] for lm in landmarks
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
    current_landmark_count = sum(
        1 for gp in grounding_papers
        if "landmark" in gp.get("relationship", "")
    )

    # FIRST: Ensure we have at least MIN_LANDMARKS landmarks
    if current_landmark_count < MIN_LANDMARKS and landmarks:
        landmarks_needed = MIN_LANDMARKS - current_landmark_count
        sorted_landmarks = sorted(
            landmarks,
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
                    "relevance": "Landmark paper in this field for context",
                })
                grounding_work_ids.add(lm["work_id"])
                landmarks_needed -= 1

    # THEN: Add refs to reach minimum grounding count
    if len(grounding_papers) < MIN_GROUNDING:
        sorted_refs = sorted(
            referenced_works,
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
                    "relevance": "Cited reference for methodology comparison",
                })
                grounding_work_ids.add(ref["work_id"])

    # FINALLY: Add more landmarks if still under minimum
    if len(grounding_papers) < MIN_GROUNDING:
        sorted_landmarks = sorted(
            landmarks,
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
                    "relevance": "Landmark paper in this field for context",
                })
                grounding_work_ids.add(lm["work_id"])

    novelty_assessment = {
        "whats_new": novelty.get("whats_new", "Unable to assess"),
        "compared_to_prior_work": novelty.get("compared_to_prior_work", "Unable to assess"),
        "novelty_level": novelty.get("novelty_level", "medium"),
        "confidence": novelty.get("confidence", "low"),
        "novelty_explanation": novelty.get("novelty_explanation", "Assessment unavailable"),
        "grounding_papers": grounding_papers[:MAX_GROUNDING],
    }

    # Validate enum values
    if novelty_assessment["novelty_level"] not in ("low", "medium", "high"):
        novelty_assessment["novelty_level"] = "medium"
    if novelty_assessment["confidence"] not in ("low", "medium", "high"):
        novelty_assessment["confidence"] = "low"

    # Check grounding consistency
    consistency = _check_grounding_consistency(
        novelty_assessment,
        referenced_works,
        landmarks,
    )

    # FIX orphaned citations: add work_ids mentioned in narrative but missing from grounding_papers
    orphaned = consistency.get("orphaned_citations", [])
    if orphaned and len(grounding_papers) < MAX_GROUNDING:
        all_papers_lookup = {r["work_id"]: r for r in referenced_works}
        all_papers_lookup.update({lm["work_id"]: lm for lm in landmarks})
        landmark_ids = {lm["work_id"] for lm in landmarks}

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
                    "relevance": "Cited in novelty assessment narrative",
                })
                grounding_work_ids.add(orphan_id)
                logger.info(f"Added orphaned citation {orphan_id} to grounding_papers")

        # Recalculate consistency after fixing orphans
        consistency = _check_grounding_consistency(
            novelty_assessment,
            referenced_works,
            landmarks,
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

    return {
        "summary": summary,
        "keywords": keywords[:10],
        "novelty_assessment": novelty_assessment,
        "_consistency_metrics": consistency,  # Internal use for monitoring
    }


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
) -> NodeDetailsResponse:
    """
    Get detailed pop-up information for a node in the citation map.

    This is the main entry point for Feature 3.
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
            logger.info(f"Enriched abstract for {work_id} from {abstract_source}")
            work_data["abstract"] = enriched_abstract
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
            logger.info(f"Inferred topic for {work_id}: {inferred_topic_id} from {topic_source}")
            work_data["primary_topic_id"] = inferred_topic_id
            enrichment_happened = True

        # Load connected works (from map edges)
        connected_works = load_connected_works(conn, map_id, work_id)

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

        # Check if we have NO grounding data at all - try LLM landmark fallback
        if len(referenced_works) == 0 and len(landmarks) == 0:
            logger.info(
                f"No grounding data for {work_id} - trying LLM landmark fallback"
            )

            # Try to get landmarks via LLM suggestion (fallback for 0+0 case)
            _, fallback_landmarks = supplement_grounding_papers(
                title=work_data["title"],
                abstract=work_data["abstract"],
                year=work_data["year"],
                existing_refs=[],
                existing_landmarks=[],
                field=work_data.get("category"),
            )

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
                    assessment_unavailable_reason=(
                        "Insufficient reference data - no citations or field landmark papers "
                        "available for comparison. Novelty assessment requires at least one "
                        "reference or landmark paper to ground the analysis."
                    ),
                )

        # Supplement grounding papers if we have some but not enough (target: 5-7)
        additional_refs, additional_landmarks = supplement_grounding_papers(
            title=work_data["title"],
            abstract=work_data["abstract"],
            year=work_data["year"],
            existing_refs=referenced_works,
            existing_landmarks=landmarks,
            field=work_data.get("category"),  # Use category as field hint
        )

        if additional_refs or additional_landmarks:
            logger.info(
                f"Supplemented grounding: +{len(additional_refs)} refs, "
                f"+{len(additional_landmarks)} landmarks"
            )
            referenced_works = referenced_works + additional_refs
            landmarks = landmarks + additional_landmarks

        # Generate via LLM with grounded context
        logger.info(f"Generating grounded node details via LLM for: {work_id}")
        llm_result = generate_node_details_llm(
            title=work_data["title"] or "Untitled",
            abstract=work_data["abstract"],
            year=work_data["year"],
            referenced_works=referenced_works,
            landmarks=landmarks,
            target_work_id=work_id,  # Prevent self-citation in grounding papers
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
            ),
            connected_works=connected_works,
        )
