"""
Timeline narrative generator for Feature 3.

Generates a rich vertical-evolution narrative for a paper's research lineage:
how the field arrived at this paper, what it changed, and what it unlocked.

NOT a methodology comparison (that's Feature 4). This tells the story over time
through one paper's lens.

Two-pass architecture:
  Pass 1: Main narrative (historical_context, contribution, impact, etc.)
  Pass 2: Dedicated era commentaries with papers pre-grouped by era
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

from app.common.id_mapping import IdMapper
from app.feature3.json_utils import extract_json_from_llm_response
from app.feature3.paper_impact_analytics import (
    _is_review_guideline_paper,
    _is_software_tool_paper,
    _truncate_text,
    calculate_impact_score,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

NARRATIVE_VERSION = "narrative-v8"
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
# Pass 1: Main narrative prompt (no era commentaries)
# ============================================================================

SYSTEM_PROMPT = """You are an expert research historian. You tell the story of how a \
research field evolved THROUGH the lens of a specific paper — what came before, what \
this paper changed, and what it unlocked afterward.

You write in DEFINITIVE prose. Never hedge. Never use these words: explores, discusses, \
examines, investigates, assesses, evaluates, addresses, looks at, studies, analyzes, reviews.

You MUST cite specific papers inline using their short IDs in brackets (e.g., [P1], [P5]). \
Always use the exact IDs shown in the paper lists.

CRITICAL CONSTRAINT: You are telling a VERTICAL EVOLUTION story — how ideas evolved over \
time in a research lineage. Do NOT compare methods side-by-side. Do NOT recommend which \
paper to use. Do NOT create strengths/weaknesses analyses. Those are methodology comparison \
tasks, not timeline narratives."""


def _format_paper_for_prompt(
    paper: Dict[str, Any], index: int, mapper: Optional[IdMapper] = None,
) -> str:
    """Format a single paper for the LLM prompt."""
    work_id = paper.get("work_id") or "?"
    display_id = mapper.add(work_id) if mapper else work_id
    title = paper.get("title") or "Untitled"
    year = paper.get("year") or "?"
    cites = paper.get("cited_by_count") or 0
    abstract = (paper.get("abstract") or "").strip()

    line = f"{index}. [{display_id}] {title} ({year}) — {cites:,} citations"
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
) -> tuple[str, IdMapper]:
    """Build the LLM prompt for main timeline narrative (pass 1, no era commentaries)."""
    year_str = f" ({year})" if year else ""
    abstract_text = abstract or "No abstract available."

    # Create mapper for all papers in this prompt
    mapper = IdMapper("P")

    # Landmarks sorted chronologically
    sorted_landmarks = sorted(landmarks[:6], key=lambda p: p.get("year") or 9999)
    landmark_lines = [
        _format_paper_for_prompt(p, i, mapper)
        for i, p in enumerate(sorted_landmarks, 1)
    ]
    landmark_section = "\n".join(landmark_lines) if landmark_lines else "(None available)"

    # References sorted chronologically
    sorted_refs = sorted(references[:15], key=lambda p: p.get("year") or 9999)
    ref_lines = [
        _format_paper_for_prompt(p, i, mapper)
        for i, p in enumerate(sorted_refs, 1)
    ]
    ref_section = "\n".join(ref_lines) if ref_lines else "(None available)"

    # Citing papers sorted chronologically
    sorted_citers = sorted(citing_papers[:20], key=lambda p: p.get("year") or 9999)
    citer_lines = [
        _format_paper_for_prompt(p, i, mapper)
        for i, p in enumerate(sorted_citers, 1)
    ]
    citer_section = "\n".join(citer_lines) if citer_lines else "(None available)"

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
spatial invariance. Cite specific papers inline using their short IDs (e.g., [P1], [P5]). Build an intellectual chain \
where each advance motivated the next.",
    "contribution_statement": "A mini paragraph (3-5 sentences) on what this paper \
specifically introduced. Name the exact mechanism (e.g., skip connections, self-attention, \
batch normalization) and explain the technical insight — WHY it works, not just WHAT it is. \
Include concrete results if notable (e.g., trained 152-layer networks, reduced error by X%).",
    "downstream_impact": "A mini paragraph (5-8 sentences) on the VERTICAL downstream \
effects within the SAME research direction. Focus on successor papers that directly \
extend, refine, scale, or apply this paper's contribution to the same problem domain. \
For each successor: name the specific architecture or technique it introduced, what \
benchmark or metric it pushed, and exactly how it built on the target paper's mechanism \
(e.g., replaced component X with Y, scaled from N to M parameters, adapted loss \
function Z). Cite specific papers inline using their short IDs (e.g., [P1], [P5]). Do NOT include cross-domain \
adoptions here — those belong in cross_domain_influence.",
    "cross_domain_influence": "A mini paragraph (3-5 sentences) on HORIZONTAL translations \
to DIFFERENT fields. For each adoption: name the specific target field, the technique \
that was adapted from this paper, what modification was required to make it work in \
the new domain (e.g., different tokenization for protein sequences, modified attention \
for graph-structured data), and the concrete result achieved. This covers ideas crossing \
disciplinary boundaries — different problem domains, different data modalities, different \
research communities. Cite specific papers inline using their short IDs (e.g., [P1], [P5]). null if not applicable.",
    "before_approach": "Dominant methodology in predecessor papers — name the specific \
technique and its key limitation (1-2 sentences)",
    "after_approach": "Dominant methodology in successor papers — name the specific \
technique and what it enabled (1-2 sentences)"
}}

RULES:
- historical_context, contribution_statement, downstream_impact, and cross_domain_influence \
(when not null) must each cite at least 2 work_ids.
- ONLY "foundational" papers can have is_paradigm_shift=true. Software, review, and \
measurement papers are NEVER paradigm shifts.
- Write in definitive prose. No hedging verbs. No "this paper explores/discusses/examines".
- Tell the evolution STORY — how each generation of work built on, reacted to, or \
departed from the previous one.
- Do NOT compare methods side-by-side or recommend which to use (that's a different feature).
- Be TECHNICAL: name specific algorithms, architectures, loss functions, metrics, \
layer counts, complexity classes. You have full paper abstracts — extract the specific \
technical details from each abstract and state them concretely in the narrative.
- BANNED PHRASES: "laid the foundation", "further enhances", "various domains", \
"various applications", "improved performance", "significant advances", \
"notable improvements", "growing interest", "increasing attention", "played a crucial \
role", "accelerated progress", "paved the way". For EVERY claim, specify: the exact \
technique name, the mechanism by which it works, and the specific domain/benchmark \
where it applies.
- Each field should read as a standalone mini paragraph, not bullet points or fragments.

Answer ONLY with the JSON object, no additional text."""

    return prompt, mapper


# ============================================================================
# Pass 2: Dedicated era commentaries
# ============================================================================

ERA_COMMENTARY_SYSTEM_PROMPT = """You write technical research narratives. Your ONLY job: \
read each paper's abstract and extract its specific technical contribution into a cohesive \
era narrative.

RULES:
- Every sentence MUST name a concrete technique, architecture, metric, or dataset
- Cite each paper using its short ID in brackets: [P1], [P2], etc.
- For each paper you mention, state WHAT it did technically: the mechanism, the numbers, \
the result — extracted directly from its abstract
- End each era narrative with the specific technical bottleneck or open problem that the \
next era addressed
- BANNED: "achieved state-of-the-art", "significant advances", "improved performance", \
"laid the foundation", "further enhances", "paved the way", "played a crucial role", \
"accelerated progress", "various domains", "notable improvements", "growing interest"

BAD (vague — DO NOT write like this):
"The era witnessed the resurgence of CNNs, with architectures like AlexNet achieving \
state-of-the-art results in image classification. The introduction of datasets like \
Microsoft COCO further accelerated progress in object detection and segmentation."

GOOD (specific mechanisms from abstracts — write like this):
"AlexNet [W2163605009] stacked five convolutional layers with ReLU activations and \
dropout regularization, training on two GPUs to classify ImageNet's 1.2M images into \
1000 classes at 37.5% top-1 error — halving the previous best. Microsoft COCO \
[W1861492603] introduced per-instance segmentation masks across 91 object categories \
with 2.5M labeled instances, enabling dense prediction benchmarks beyond classification. \
The remaining bottleneck was network depth: beyond ~20 layers, gradient degradation \
prevented convergence."

You MUST write at this level of technical specificity. If the abstract says the method \
uses "a novel attention mechanism", you must say what KIND of attention, how it differs, \
and what metric it achieved."""


def _build_era_commentary_prompt(
    target_title: str,
    era_papers: Dict[str, List[Dict[str, Any]]],
    era_labels: List[str],
    mapper: Optional[IdMapper] = None,
) -> tuple[str, IdMapper]:
    """Build a focused prompt for era commentary generation (pass 2).

    Papers are pre-grouped by era so the LLM doesn't have to figure out grouping.
    """
    if mapper is None:
        mapper = IdMapper("P")
    sections = []
    for era in era_labels:
        papers = era_papers.get(era, [])
        if not papers:
            continue
        paper_lines = []
        for i, p in enumerate(papers, 1):
            work_id = p.get("work_id") or "?"
            display_id = mapper.add(work_id)
            title = p.get("title") or "Untitled"
            year = p.get("year") or "?"
            cites = p.get("cited_by_count") or 0
            abstract = (p.get("abstract") or "").strip()
            line = f"  {i}. [{display_id}] {title} ({year}) — {cites:,} citations"
            if abstract:
                line += f"\n     ABSTRACT: {abstract}"
            else:
                line += "\n     ABSTRACT: (not available)"
            paper_lines.append(line)
        sections.append(f"### Era: {era}\n" + "\n".join(paper_lines))

    papers_text = "\n\n".join(sections)

    era_json_examples = ", ".join(f'"{e}"' for e in era_labels)

    prompt = f"""## CONTEXT
Target paper: {target_title}

## PAPERS GROUPED BY ERA (read each abstract carefully)

{papers_text}

---

For each era above, write a technical narrative paragraph. Extract specific details \
from each paper's abstract: the architecture, the mechanism, the metric, the dataset, \
the result. Do NOT summarize vaguely.

Return a JSON array with one object per era:
[
    {{
        "era": "{era_labels[0] if era_labels else '2020s'}",
        "headline": "Short descriptive title for this era (e.g., 'Denoising diffusion emergence')",
        "narrative": "Technical narrative paragraph. For each paper: state its short ID in \
brackets (e.g., [P1]), then what it specifically did (architecture, loss function, training procedure, \
benchmark result). End with the bottleneck the next era solved.",
        "key_work_ids": ["P1", "P3"]
    }}
]

REQUIREMENTS:
- One entry per era: [{era_json_examples}]
- Each narrative must cite every paper from that era by its short [ID]
- Extract technical details FROM THE ABSTRACTS — do not invent claims
- key_work_ids must list the work_ids actually cited in the narrative

Answer ONLY with the JSON array, no additional text."""

    return prompt, mapper


def _call_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    expected_type: str = "object",
) -> Optional[Dict[str, Any] | List[Any]]:
    """Make an LLM call with retries. Returns parsed JSON or None."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=90.0,
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip()

            result, error = extract_json_from_llm_response(content, expected_type=expected_type)
            if result is None:
                logger.warning(f"Failed to parse LLM JSON: {error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
                return None

            return result

        except Exception as e:
            logger.warning(f"LLM call exception: {e}")
            error_str = str(e).lower()
            is_transient = (
                "rate" in error_str
                or "timeout" in error_str
                or "connection" in error_str
            )
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            return None

    return None


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
    era_papers: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate a rich timeline narrative for a paper's research lineage.

    Two-pass architecture:
      Pass 1: Main narrative (historical_context, contribution, impact)
      Pass 2: Dedicated era commentaries with papers pre-grouped by era

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

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, cannot generate timeline narrative")
        return None

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    # ── Pass 1: Main narrative ──────────────────────────────────────────
    prompt, mapper = _build_narrative_prompt(
        title, abstract, year, cited_by_count,
        references, landmarks, citing_papers,
    )

    result = _call_llm(client, SYSTEM_PROMPT, prompt, expected_type="object")
    if result is None:
        logger.warning(f"Pass 1 (main narrative) failed for {work_id}")
        return None

    # Resolve short IDs back to real work_ids in narrative text
    for field in ("historical_context", "contribution_statement",
                  "downstream_impact", "cross_domain_influence",
                  "before_approach", "after_approach"):
        if result.get(field):
            result[field] = mapper.resolve_text(result[field])

    # ── Pass 2: Era commentaries ────────────────────────────────────────
    if era_labels and era_papers:
        logger.info(f"Pass 2: generating era commentaries for {len(era_labels)} eras")
        era_prompt, era_mapper = _build_era_commentary_prompt(
            title, era_papers, era_labels, mapper,  # reuse mapper from pass 1
        )
        era_result = _call_llm(
            client, ERA_COMMENTARY_SYSTEM_PROMPT, era_prompt, expected_type="array",
        )
        if era_result and isinstance(era_result, list):
            # Resolve short IDs in era commentaries
            for ec in era_result:
                if ec.get("narrative"):
                    ec["narrative"] = era_mapper.resolve_text(ec["narrative"])
                if ec.get("key_work_ids"):
                    ec["key_work_ids"] = era_mapper.resolve_list(ec["key_work_ids"])
            result["era_commentaries"] = era_result
            logger.info(f"Era commentaries generated: {len(era_result)} eras")
        else:
            logger.warning(f"Pass 2 (era commentaries) failed for {work_id}, using empty")
            result["era_commentaries"] = []
    else:
        result["era_commentaries"] = []

    # ── Post-process ────────────────────────────────────────────────────
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
