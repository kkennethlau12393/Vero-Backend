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

from app.feature3.json_utils import extract_json_from_llm_response
from app.feature3.paper_impact_analytics import (
    _is_review_guideline_paper,
    _is_software_tool_paper,
    _truncate_text,
    calculate_impact_score,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

NARRATIVE_VERSION = "narrative-v12"
MODEL_VERSION = "openai/gpt-oss-120b"
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

You MUST cite specific work_ids inline using their FULL IDs in brackets. \
OpenAlex papers use [W2163605009], Semantic Scholar papers use [S2:204e3073870f], \
and ArXiv papers use [AX:2301.12345]. NEVER truncate or abbreviate work_ids — \
always write the complete ID.

CRITICAL CONSTRAINT: You are telling a VERTICAL EVOLUTION story — how ideas evolved over \
time in a research lineage. Do NOT compare methods side-by-side. Do NOT recommend which \
paper to use. Do NOT create strengths/weaknesses analyses. Those are methodology comparison \
tasks, not timeline narratives.

CITATION STYLE: When citing a paper, NEVER write its title in the prose. Use the citation \
marker [work_id] AS the subject — the UI renders it as a clickable chip that already shows \
the title.
BAD: "Crenshaw's Mapping the Margins [W2163605009] exposed..."
GOOD: "[W2163605009] exposed the analytical blind spot..."
GOOD: "Crenshaw's foundational work [W2163605009] exposed..." """


def _format_paper_for_prompt(
    paper: Dict[str, Any], index: int,
) -> str:
    """Format a single paper for the LLM prompt."""
    work_id = paper.get("work_id") or "?"
    title = paper.get("title") or "Untitled"
    year = paper.get("year") or "?"
    cites = paper.get("cited_by_count") or 0
    abstract = (paper.get("abstract") or "").strip()

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
) -> str:
    """Build the LLM prompt for main timeline narrative (pass 1, no era commentaries)."""
    year_str = f" ({year})" if year else ""
    abstract_text = abstract or "No abstract available."

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
spatial invariance. Cite specific work_ids inline (e.g., [W2163605009]). Build an intellectual chain \
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
function Z). Cite specific work_ids inline (e.g., [W2163605009]). Do NOT include cross-domain \
adoptions here — those belong in cross_domain_influence.",
    "cross_domain_influence": "A mini paragraph (3-5 sentences) on HORIZONTAL translations \
to DIFFERENT fields. For each adoption: name the specific target field, the technique \
that was adapted from this paper, what modification was required to make it work in \
the new domain (e.g., different tokenization for protein sequences, modified attention \
for graph-structured data), and the concrete result achieved. This covers ideas crossing \
disciplinary boundaries — different problem domains, different data modalities, different \
research communities. Cite specific work_ids inline (e.g., [W2163605009]). null if not applicable.",
    "before_approach": "Dominant methodology in predecessor papers — name the specific \
technique and its key limitation (1-2 sentences)",
    "after_approach": "Dominant methodology in successor papers — name the specific \
technique and what it enabled (1-2 sentences)"
}}

RULES:
- NEVER write a paper's title in the prose. The [work_id] renders as a clickable chip \
showing the title. Use the marker as the subject or after a brief descriptor.
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

    return prompt


# ============================================================================
# Pass 2: Dedicated era commentaries
# ============================================================================

ERA_COMMENTARY_SYSTEM_PROMPT = """You write technical research narratives. Your ONLY job: \
read each paper's abstract and extract its specific technical contribution into a cohesive \
era narrative broken into thematic subsections.

RULES:
- Every sentence MUST name a concrete technique, architecture, metric, or dataset
- Cite each paper using its FULL work_id in brackets: [W2163605009], [S2:204e3073870f], or [AX:2301.12345]. NEVER truncate.
- NEVER write a paper's title in the prose — the [work_id] renders as a clickable chip showing the title. Use the marker as the subject: "[W2163605009] stacked five convolutional layers..."
- For each paper you mention, state WHAT it did technically: the mechanism, the numbers, \
the result — extracted directly from its abstract
- Break each era into 2-4 thematic subsections. Each subsection has a short heading \
(e.g., "Architecture innovations", "Training paradigm shifts") and a body paragraph \
covering 1-3 papers that share that theme.
- The LAST subsection of each era should end with the specific technical bottleneck or \
open problem that the next era addressed.
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
) -> str:
    """Build a focused prompt for era commentary generation (pass 2).

    Papers are pre-grouped by era so the LLM doesn't have to figure out grouping.
    """
    sections = []
    for era in era_labels:
        papers = era_papers.get(era, [])
        if not papers:
            continue
        paper_lines = []
        for i, p in enumerate(papers, 1):
            work_id = p.get("work_id") or "?"
            title = p.get("title") or "Untitled"
            year = p.get("year") or "?"
            cites = p.get("cited_by_count") or 0
            abstract = (p.get("abstract") or "").strip()
            line = f"  {i}. [{work_id}] {title} ({year}) — {cites:,} citations"
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

For each era above, break the narrative into 2-4 thematic subsections. Each subsection \
groups 1-3 papers that share a theme (e.g., "Architecture innovations", "Benchmark \
datasets", "Training paradigm shifts"). Extract specific details from each paper's \
abstract: the architecture, the mechanism, the metric, the dataset, the result. \
Do NOT summarize vaguely.

Return a JSON array with one object per era:
[
    {{
        "era": "{era_labels[0] if era_labels else '2020s'}",
        "headline": "Short descriptive title for this era (e.g., 'Denoising diffusion emergence')",
        "subsections": [
            {{
                "heading": "Short thematic heading (e.g., 'Architecture innovations')",
                "body": "Technical paragraph. For each paper: state its work_id in \
brackets, then what it specifically did (architecture, loss function, training procedure, \
benchmark result)."
            }}
        ],
        "key_work_ids": ["W...", "S..."]
    }}
]

REQUIREMENTS:
- One entry per era: [{era_json_examples}]
- 2-4 subsections per era, each with a heading and body
- Each subsection body must cite every paper it covers by [work_id]
- The last subsection should end with the bottleneck the next era solved
- Extract technical details FROM THE ABSTRACTS — do not invent claims
- If a paper has "(not available)" as its abstract, ONLY state its title and citation \
count. Do NOT fabricate methods, results, or mechanisms for papers without abstracts
- key_work_ids must list ALL work_ids actually cited across all subsections

Answer ONLY with the JSON array, no additional text."""

    return prompt


def _call_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    expected_type: str = "object",
    use_json_mode: bool = False,
) -> Optional[Dict[str, Any] | List[Any]]:
    """Make an LLM call with retries. Returns parsed JSON or None."""
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=90.0,
                temperature=0,
                max_tokens=8192,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            finish_reason = resp.choices[0].finish_reason
            if finish_reason != "stop":
                logger.warning(f"LLM finish_reason={finish_reason} (expected 'stop')")

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
                or "json_validate" in error_str
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

    def _scrub_field(val):
        """Scrub a field that may be a raw string or a StructuredText dict."""
        if isinstance(val, str):
            return _scrub_banned_verbs(val)
        elif isinstance(val, dict) and "text" in val:
            val["text"] = _scrub_banned_verbs(val["text"])
            return val
        return val

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
        if val:
            narrative[field] = _scrub_field(val)

    for ec in narrative.get("era_commentaries", []):
        if ec.get("narrative"):
            ec["narrative"] = _scrub_field(ec["narrative"])
        if ec.get("headline"):
            ec["headline"] = _scrub_field(ec["headline"])
        for sub in ec.get("subsections", []):
            if sub.get("body"):
                sub["body"] = _scrub_field(sub["body"])
            if sub.get("heading"):
                sub["heading"] = _scrub_field(sub["heading"])

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
# Structured Citations
# ============================================================================

def _structure_citations(
    text: str,
    paper_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert raw LLM text with [W...] citations into structured format.

    Replaces [W2163605009] with (N) and builds a citations array.
    """
    import re

    ID_PAT = r'\[(W\d+|S2:[a-fA-F0-9]+|S[a-fA-F0-9]{20,}|AX:[\d.]+)\]'
    citations: List[Dict[str, Any]] = []
    ref_map: Dict[str, int] = {}

    def replacer(match):
        work_id = match.group(1)
        if work_id in ref_map:
            return f"({ref_map[work_id]})"

        ref_num = len(citations) + 1
        ref_map[work_id] = ref_num

        paper = paper_lookup.get(work_id, {})
        citations.append({
            "ref": ref_num,
            "work_id": work_id,
            "title": paper.get("title"),
            "year": paper.get("year"),
            "cited_by_count": paper.get("cited_by_count"),
            "authors": paper.get("authors", []),
        })
        return f"({ref_num})"

    structured_text = re.sub(ID_PAT, replacer, text)
    return {"text": structured_text, "citations": citations}


def _structure_citations_shared(
    text: str,
    paper_lookup: Dict[str, Dict[str, Any]],
    ref_map: Dict[str, int],
    citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Like _structure_citations but shares ref_map and citations list across calls.

    This keeps numbering continuous across multiple text blocks (e.g. subsections).
    """
    import re

    ID_PAT = r'\[(W\d+|S2:[a-fA-F0-9]+|S[a-fA-F0-9]{20,}|AX:[\d.]+)\]'

    def replacer(match):
        work_id = match.group(1)
        if work_id in ref_map:
            return f"({ref_map[work_id]})"

        ref_num = len(citations) + 1
        ref_map[work_id] = ref_num

        paper = paper_lookup.get(work_id, {})
        citations.append({
            "ref": ref_num,
            "work_id": work_id,
            "title": paper.get("title"),
            "year": paper.get("year"),
            "cited_by_count": paper.get("cited_by_count"),
            "authors": paper.get("authors", []),
        })
        return f"({ref_num})"

    structured_text = re.sub(ID_PAT, replacer, text)
    return {"text": structured_text, "citations": citations}


# ============================================================================
# Technical Term Extraction
# ============================================================================

TERM_EXTRACTION_SYSTEM_PROMPT = """You extract technical terms from research narratives and provide \
concise explanations. Each explanation should be 1-2 sentences, technically precise but accessible \
to someone outside the specific subfield. Focus on mechanisms, not definitions."""


def _build_term_extraction_prompt(narrative_text: str) -> str:
    return f"""Extract the key technical terms from this research narrative. For each term, \
provide a 1-2 sentence explanation of what it IS and HOW it works (mechanism, not just definition).

Only extract terms that are:
- Specific techniques, architectures, or algorithms (not generic words like "model" or "approach")
- Would benefit from explanation for someone in a related but different field

Text:
{narrative_text}

Return a JSON object with a "terms" key:
{{
    "terms": [
        {{"term": "self-attention", "explanation": "A mechanism where..."}},
        ...
    ]
}}

Max 10 terms. Answer ONLY with the JSON object."""


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

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, cannot generate timeline narrative")
        return None

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    # ── Pass 1: Main narrative ──────────────────────────────────────────
    prompt = _build_narrative_prompt(
        title, abstract, year, cited_by_count,
        references, landmarks, citing_papers,
    )

    result = _call_llm(client, SYSTEM_PROMPT, prompt, expected_type="object", use_json_mode=True)
    if result is None:
        logger.warning(f"Pass 1 (main narrative) failed for {work_id}")
        return None

    # Calculate impact score (needs is_paradigm_shift from Pass 1)
    impact_result = calculate_impact_score(
        cited_by_count, references, result.get("is_paradigm_shift", False)
    )

    # ── Pass 2: Era commentaries ────────────────────────────────────────
    if era_labels and era_papers:
        # Check which eras have at least one paper with an abstract
        eras_with_abstracts = []
        eras_without_abstracts = []
        for era in era_labels:
            papers = era_papers.get(era, [])
            has_any_abstract = any(
                (p.get("abstract") or "").strip() for p in papers
            )
            if has_any_abstract:
                eras_with_abstracts.append(era)
            else:
                eras_without_abstracts.append(era)

        if eras_without_abstracts:
            logger.info(f"Skipping era commentary for {eras_without_abstracts} (no abstracts available)")

        # Only send eras with abstracts to LLM
        filtered_era_labels = eras_with_abstracts
        filtered_era_papers = {e: era_papers[e] for e in eras_with_abstracts if e in era_papers}

        # Build placeholder commentaries for abstract-less eras
        placeholder_eras = []
        for era in eras_without_abstracts:
            papers = era_papers.get(era, [])
            paper_titles = [p.get("title", "Untitled") for p in papers]
            work_ids = [p.get("work_id") for p in papers if p.get("work_id")]
            placeholder_eras.append({
                "era": era,
                "headline": f"Early foundations ({len(papers)} paper{'s' if len(papers) != 1 else ''})",
                "narrative": f"Abstracts are not available for papers in this era. "
                    f"Based on titles alone: {'; '.join(paper_titles[:5])}. "
                    f"Detailed analysis requires access to full paper text.",
                "subsections": [],
                "key_work_ids": work_ids,
            })

        if filtered_era_labels:
            logger.info(f"Pass 2: generating era commentaries for {len(filtered_era_labels)} eras (skipping {len(eras_without_abstracts)} without abstracts)")
            era_prompt = _build_era_commentary_prompt(title, filtered_era_papers, filtered_era_labels)
            era_result = _call_llm(
                client, ERA_COMMENTARY_SYSTEM_PROMPT, era_prompt, expected_type="array",
            )
            if era_result and isinstance(era_result, list):
                # Build flat narrative fallback from subsections (BEFORE structured citation conversion)
                for ec in era_result:
                    subs = ec.get("subsections", [])
                    if subs and not ec.get("narrative"):
                        raw_bodies = [s.get("body", "") for s in subs if isinstance(s.get("body"), str)]
                        ec["narrative"] = " ".join(raw_bodies) if raw_bodies else ""
                    elif not ec.get("narrative"):
                        ec["narrative"] = ""
                # Combine: placeholders first (older eras), then LLM-generated
                all_eras = placeholder_eras + era_result
                # Sort by era label to maintain chronological order
                era_order = {e: i for i, e in enumerate(era_labels)}
                all_eras.sort(key=lambda x: era_order.get(x.get("era", ""), 999))
                result["era_commentaries"] = all_eras
                logger.info(f"Era commentaries: {len(era_result)} generated + {len(placeholder_eras)} placeholders")
            else:
                logger.warning(f"Pass 2 (era commentaries) failed for {work_id}, using placeholders only")
                result["era_commentaries"] = placeholder_eras
        else:
            # All eras lack abstracts
            result["era_commentaries"] = placeholder_eras
            logger.info(f"All {len(eras_without_abstracts)} eras lack abstracts, using placeholders only")
    else:
        result["era_commentaries"] = []

    # ── Post-process ────────────────────────────────────────────────────
    result = _enforce_paper_type_constraints(result, title, abstract)
    result = _scrub_narrative_verbs(result)
    result["impact_score"] = impact_result["overall"]
    result["impact_breakdown"] = impact_result

    # Validate work_id citations
    known_ids = set()
    for paper_list in [references, landmarks, citing_papers]:
        for p in paper_list:
            wid = p.get("work_id")
            if wid:
                known_ids.add(wid)
    _validate_work_id_citations(result, known_ids)

    # ── Pass 3: Technical term extraction ─────────────────────────────
    all_narrative_text = " ".join(filter(None, [
        result.get("contribution_statement") if isinstance(result.get("contribution_statement"), str)
            else (result.get("contribution_statement") or {}).get("text"),
        result.get("downstream_impact") if isinstance(result.get("downstream_impact"), str)
            else (result.get("downstream_impact") or {}).get("text"),
        *[ec.get("narrative") if isinstance(ec.get("narrative"), str)
            else (ec.get("narrative") or {}).get("text", "")
          for ec in result.get("era_commentaries", [])],
    ]))

    if all_narrative_text:
        term_prompt = _build_term_extraction_prompt(all_narrative_text[:3000])
        terms = _call_llm(client, TERM_EXTRACTION_SYSTEM_PROMPT, term_prompt, expected_type="object", use_json_mode=True)
        logger.info(f"Pass 3 (technical terms) raw result type={type(terms).__name__}, value={str(terms)[:200]}")
        # Unwrap: JSON mode may return {"terms": [...]} or just a list
        if terms and isinstance(terms, dict):
            terms = terms.get("terms", [])
        elif terms and isinstance(terms, list):
            pass  # Already a list
        else:
            terms = []
        # Validate each term has required fields
        valid_terms = []
        for t in terms:
            if isinstance(t, dict) and t.get("term") and t.get("explanation"):
                valid_terms.append({"term": t["term"], "explanation": t["explanation"]})
        result["technical_terms"] = valid_terms
        logger.info(f"Pass 3: extracted {len(valid_terms)} technical terms")
    else:
        logger.info("Pass 3: skipped (no narrative text)")
        result["technical_terms"] = []

    # ── Structured citations ──────────────────────────────────────────
    # Build paper lookup for citation metadata
    paper_lookup: Dict[str, Dict[str, Any]] = {}
    for paper_list in [references, landmarks, citing_papers]:
        for p in paper_list:
            wid = p.get("work_id")
            if wid and wid not in paper_lookup:
                paper_lookup[wid] = {
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "cited_by_count": p.get("cited_by_count"),
                    "authors": p.get("authors", []),
                }

    # Convert narrative fields to structured format
    for field in ["contribution_statement", "downstream_impact", "cross_domain_influence",
                   "historical_context"]:
        val = result.get(field)
        if val and isinstance(val, str):
            result[field] = _structure_citations(val, paper_lookup)

    # Convert era commentary fields with continuous numbering per era
    for ec in result.get("era_commentaries", []):
        # Shared state for this era — numbering is continuous across narrative + all subsections
        era_ref_map: Dict[str, int] = {}
        era_citations: List[Dict[str, Any]] = []

        if ec.get("narrative") and isinstance(ec["narrative"], str):
            ec["narrative"] = _structure_citations_shared(ec["narrative"], paper_lookup, era_ref_map, era_citations)
        for sub in ec.get("subsections", []):
            if sub.get("body") and isinstance(sub["body"], str):
                sub["body"] = _structure_citations_shared(sub["body"], paper_lookup, era_ref_map, era_citations)

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
