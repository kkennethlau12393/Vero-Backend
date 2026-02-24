"""
Timeline narrative generator for Feature 3.

Generates a rich vertical-evolution narrative for a paper's research lineage:
how the field arrived at this paper, what it changed, and what it unlocked.

NOT a methodology comparison (that's Feature 4). This tells the story over time
through one paper's lens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.feature3.json_utils import extract_json_from_llm_response
from app.feature3.paper_impact_analytics import (
    _is_review_guideline_paper,
    _is_software_tool_paper,
    _truncate_text,
    calculate_impact_score,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

NARRATIVE_VERSION = "narrative-v3"
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5


# ============================================================================
# Cache
# ============================================================================

def _get_cached_narrative(conn: Connection, work_id: str) -> Optional[Dict[str, Any]]:
    """Read cached timeline narrative, version-gated."""
    row = conn.execute(
        text("""
            SELECT narrative_json FROM timeline_narrative_cache
            WHERE work_id = :work_id AND narrative_version = :version
        """),
        {"work_id": work_id, "version": NARRATIVE_VERSION},
    ).mappings().first()
    if row:
        return row["narrative_json"]
    return None


def _cache_narrative(conn: Connection, work_id: str, narrative: Dict[str, Any]) -> None:
    """Store timeline narrative in cache."""
    conn.execute(
        text("""
            INSERT INTO timeline_narrative_cache (work_id, narrative_json, narrative_version)
            VALUES (:work_id, CAST(:narrative AS jsonb), :version)
            ON CONFLICT (work_id) DO UPDATE SET
                narrative_json = CAST(EXCLUDED.narrative_json AS jsonb),
                narrative_version = EXCLUDED.narrative_version,
                created_at = now()
        """),
        {"work_id": work_id, "narrative": json.dumps(narrative), "version": NARRATIVE_VERSION},
    )
    conn.commit()


# ============================================================================
# Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an expert research historian. You tell the story of how a \
research field evolved THROUGH the lens of a specific paper — what came before, what \
this paper changed, and what it unlocked afterward.

You write in DEFINITIVE prose. Never hedge. Never use these words: explores, discusses, \
examines, investigates, assesses, evaluates, addresses, looks at, studies, analyzes, reviews.

You MUST cite specific work_ids inline using their exact IDs in brackets. \
OpenAlex papers use [W...] (e.g., [W2163605009]), Semantic Scholar papers use [S...] \
(e.g., [S1234abcd]), and ArXiv papers use [AX...] (e.g., [AX2301.12345]).

CRITICAL CONSTRAINT: You are telling a VERTICAL EVOLUTION story — how ideas evolved over \
time in a research lineage. Do NOT compare methods side-by-side. Do NOT recommend which \
paper to use. Do NOT create strengths/weaknesses analyses. Those are methodology comparison \
tasks, not timeline narratives."""


def _format_paper_for_prompt(
    paper: Dict[str, Any], index: int, abstract_max: int = 300
) -> str:
    """Format a single paper for the LLM prompt."""
    work_id = paper.get("work_id") or "?"
    title = paper.get("title") or "Untitled"
    year = paper.get("year") or "?"
    cites = paper.get("cited_by_count") or 0
    abstract = _truncate_text(paper.get("abstract") or "", abstract_max)

    line = f"{index}. [{work_id}] {title} ({year}) — {cites:,} citations"
    if abstract:
        line += f"\n   {abstract}"
    return line


def _build_narrative_prompt(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    cited_by_count: int,
    references: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    citing_papers: List[Dict[str, Any]],
    era_labels: Optional[List[str]] = None,
) -> str:
    """Build the LLM prompt for timeline narrative generation."""
    year_str = f" ({year})" if year else ""
    abstract_text = _truncate_text(abstract or "No abstract available.", 500)

    # Landmarks sorted chronologically
    sorted_landmarks = sorted(landmarks[:6], key=lambda p: p.get("year") or 9999)
    landmark_lines = [
        _format_paper_for_prompt(p, i)
        for i, p in enumerate(sorted_landmarks, 1)
    ]
    landmark_section = "\n".join(landmark_lines) if landmark_lines else "(None available)"

    # References sorted chronologically
    sorted_refs = sorted(references[:15], key=lambda p: p.get("year") or 9999)
    ref_lines = [
        _format_paper_for_prompt(p, i)
        for i, p in enumerate(sorted_refs, 1)
    ]
    ref_section = "\n".join(ref_lines) if ref_lines else "(None available)"

    # Citing papers sorted chronologically
    sorted_citers = sorted(citing_papers[:20], key=lambda p: p.get("year") or 9999)
    citer_lines = [
        _format_paper_for_prompt(p, i)
        for i, p in enumerate(sorted_citers, 1)
    ]
    citer_section = "\n".join(citer_lines) if citer_lines else "(None available)"

    # Build era labels section for the prompt
    if era_labels and len(era_labels) > 0:
        era_example = era_labels[0]
        era_labels_str = ", ".join(f'"{e}"' for e in era_labels)
        era_instruction = (
            f"- era_commentaries MUST use EXACTLY these era labels: [{era_labels_str}]. "
            f"One commentary per era label. Do not invent new era labels."
        )
    else:
        era_example = "1990s"
        era_instruction = (
            "- era_commentaries MUST cover every time period that has papers above, "
            "from the earliest predecessor to the latest successor. "
            "Include the target paper's own era."
        )

    prompt = f"""## TARGET PAPER
Title: {title}{year_str}
Citations: {cited_by_count:,}
Abstract: {abstract_text}

## PREDECESSOR PAPERS (sorted chronologically)

### Landmarks (foundational works in this field)
{landmark_section}

### Direct References (papers this work cites)
{ref_section}

## SUCCESSOR PAPERS (papers that cite this work, sorted chronologically)
{citer_section}

---

Analyze this paper's place in its research lineage. Return JSON:
{{
    "paper_type": "software" | "review" | "foundational" | "empirical" | "measurement",
    "is_paradigm_shift": true | false,
    "historical_context": "A mini paragraph (5-8 sentences) telling how the field arrived \
at this paper. Name specific algorithms, architectures, loss functions, or theoretical \
frameworks that predecessors introduced. Explain what each solved and what concrete \
limitation remained — e.g., vanishing gradients at N layers, O(n^2) complexity, lack of \
spatial invariance. Cite specific [W...] work_ids inline. Build an intellectual chain \
where each advance motivated the next.",
    "contribution_statement": "A mini paragraph (3-5 sentences) on what this paper \
specifically introduced. Name the exact mechanism (e.g., skip connections, self-attention, \
batch normalization) and explain the technical insight — WHY it works, not just WHAT it is. \
Include concrete results if notable (e.g., trained 152-layer networks, reduced error by X%).",
    "downstream_impact": "A mini paragraph (5-8 sentences) on what new research directions \
this paper opened. Name specific techniques, architectures, or applications that successors \
built. Explain how they extended, adapted, or combined the contribution with other ideas. \
Cite specific [W...] work_ids inline. Cover both direct extensions and unexpected \
applications in other domains.",
    "era_commentaries": [
        {{
            "era": "{era_example}",
            "headline": "Short era title (e.g., 'The statistical learning era')",
            "narrative": "A technical mini paragraph (4-6 sentences) about this era's \
contribution to the research lineage. Name specific algorithms, architectures, or \
theoretical results. Explain what technical barrier was hit and why the field moved on. \
Cite [W...] work_ids. Connect this era to the next — what question or limitation \
motivated the transition.",
            "key_work_ids": ["W123", "W456"]
        }}
    ],
    "cross_domain_influence": "2-3 sentences naming specific fields and applications \
where this paper's ideas were adopted (e.g., NLP transformers applied to protein \
folding, GANs used in drug discovery). null if not applicable.",
    "before_approach": "Dominant methodology in predecessor papers — name the specific \
technique and its key limitation (1-2 sentences)",
    "after_approach": "Dominant methodology in successor papers — name the specific \
technique and what it enabled (1-2 sentences)"
}}

RULES:
{era_instruction}
- Each era narrative must cite at least one work_id from that era (e.g., [W...], [S...], or [AX...]).
- historical_context, contribution_statement, and downstream_impact must each cite at \
least 3 work_ids.
- ONLY "foundational" papers can have is_paradigm_shift=true. Software, review, and \
measurement papers are NEVER paradigm shifts.
- Write in definitive prose. No hedging verbs. No "this paper explores/discusses/examines".
- Tell the evolution STORY — how each generation of work built on, reacted to, or \
departed from the previous one.
- Do NOT compare methods side-by-side or recommend which to use (that's a different feature).
- Be TECHNICAL: name specific algorithms, architectures, loss functions, metrics, \
layer counts, complexity classes. Avoid vague statements like "improved performance" — \
say HOW and by what mechanism.
- Each field should read as a standalone mini paragraph, not bullet points or fragments.

Answer ONLY with the JSON object, no additional text."""

    return prompt


# ============================================================================
# Post-processing
# ============================================================================

def _scrub_narrative_verbs(narrative: Dict[str, Any]) -> Dict[str, Any]:
    """Scrub banned verbs from all narrative text fields."""
    from app.feature3.node_details_service import _scrub_banned_verbs

    text_fields = [
        "historical_context",
        "contribution_statement",
        "downstream_impact",
        "cross_domain_influence",
        "before_approach",
        "after_approach",
    ]
    for field in text_fields:
        val = narrative.get(field)
        if val and isinstance(val, str):
            narrative[field] = _scrub_banned_verbs(val)

    for ec in narrative.get("era_commentaries", []):
        if ec.get("narrative"):
            ec["narrative"] = _scrub_banned_verbs(ec["narrative"])
        if ec.get("headline"):
            ec["headline"] = _scrub_banned_verbs(ec["headline"])

    return narrative


def _validate_work_id_citations(
    narrative: Dict[str, Any],
    known_work_ids: set,
) -> None:
    """Log warnings for work_ids cited in narrative but not in the provided paper lists."""
    all_text = " ".join(
        str(narrative.get(f) or "")
        for f in [
            "historical_context",
            "contribution_statement",
            "downstream_impact",
            "cross_domain_influence",
        ]
    )
    for ec in narrative.get("era_commentaries", []):
        all_text += " " + (ec.get("narrative") or "")

    # Match all work_id formats: W... (OpenAlex), S... (S2), AX... (ArXiv)
    cited_ids = set(re.findall(r"(?:W\d{8,}|S[a-f0-9]{10,}|AX[\d.]+)", all_text))
    unknown = cited_ids - known_work_ids
    if unknown:
        logger.warning(
            f"Timeline narrative cites {len(unknown)} unknown work_ids "
            f"(possible hallucinations): {unknown}"
        )


def _enforce_paper_type_constraints(
    narrative: Dict[str, Any],
    title: Optional[str],
    abstract: Optional[str],
) -> Dict[str, Any]:
    """Enforce paradigm shift constraints based on paper type."""
    paper_type = narrative.get("paper_type", "foundational")

    # Pattern-based overrides (more reliable than LLM)
    if _is_software_tool_paper(title, abstract):
        paper_type = "software"
        narrative["paper_type"] = "software"
    elif _is_review_guideline_paper(title, abstract):
        paper_type = "review"
        narrative["paper_type"] = "review"

    # Only foundational papers can be paradigm shifts
    if paper_type in ("software", "review", "measurement"):
        if narrative.get("is_paradigm_shift"):
            logger.info(
                f"Overriding paradigm_shift to false for {paper_type} paper"
            )
            narrative["is_paradigm_shift"] = False

    return narrative


# ============================================================================
# Main
# ============================================================================

def generate_timeline_narrative(
    conn: Connection,
    work_id: str,
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    cited_by_count: int,
    references: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
    citing_papers: List[Dict[str, Any]],
    era_labels: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a rich timeline narrative for a paper's research lineage.

    Returns dict matching ResearchLineageNarrative schema, or None on failure.
    """
    # Check cache first
    try:
        cached = _get_cached_narrative(conn, work_id)
        if cached:
            logger.info(f"Timeline narrative cache hit for {work_id}")
            return cached
    except Exception as e:
        logger.warning(f"Cache read failed for {work_id}: {e}")

    # Need at least some papers to build a narrative
    if not references and not landmarks and not citing_papers:
        logger.info(f"No papers available for timeline narrative of {work_id}")
        return None

    # Calculate impact score (deterministic, no LLM needed)
    impact_score = calculate_impact_score(cited_by_count, references)

    # Build prompt
    prompt = _build_narrative_prompt(
        title, abstract, year, cited_by_count,
        references, landmarks, citing_papers,
        era_labels=era_labels,
    )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, cannot generate timeline narrative")
        return None

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                timeout=90.0,
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()

            result, error = extract_json_from_llm_response(content, expected_type="object")
            if result is None:
                logger.warning(f"Failed to parse timeline narrative JSON: {error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
                break

            # Post-process
            result = _enforce_paper_type_constraints(result, title, abstract)
            result = _scrub_narrative_verbs(result)
            result["impact_score"] = impact_score

            # Validate work_id citations
            known_ids = set()
            for paper_list in [references, landmarks, citing_papers]:
                for p in paper_list:
                    wid = p.get("work_id")
                    if wid:
                        known_ids.add(wid)
            _validate_work_id_citations(result, known_ids)

            logger.info(
                f"Timeline narrative generated for {work_id}: "
                f"type={result.get('paper_type')}, "
                f"paradigm_shift={result.get('is_paradigm_shift')}, "
                f"eras={len(result.get('era_commentaries', []))}"
            )

            # Cache the result
            try:
                _cache_narrative(conn, work_id, result)
            except Exception as e:
                logger.warning(f"Cache write failed for {work_id}: {e}")

            return result

        except Exception as e:
            logger.warning(f"Timeline narrative LLM call exception: {e}")
            error_str = str(e).lower()
            is_transient = (
                "rate" in error_str
                or "timeout" in error_str
                or "connection" in error_str
            )
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            break

    logger.warning(f"Timeline narrative generation failed for {work_id}")
    return None
