"""
Temporal analytics for Feature 2.

This module detects breakthrough years and evolution trends across
temporal maps, using citation patterns and LLM analysis.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.engine import Connection

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _truncate_text(text_val: str, max_chars: int = 200) -> str:
    """Truncate text to max characters."""
    if not text_val:
        return ""
    if len(text_val) <= max_chars:
        return text_val
    return text_val[:max_chars] + "..."


def detect_breakthrough_era(
    era_data: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Detect the breakthrough era based on citation velocity spikes.

    An era is a breakthrough era if:
    1. Its average citation count is significantly higher than the overall average
    2. It contains multiple milestone papers

    Returns dict with breakthrough era info or None if no clear breakthrough.
    """
    if len(era_data) < 2:
        return None

    # Calculate average citations per era
    era_stats = []
    total_citations = 0
    total_papers = 0

    for era, data in era_data.items():
        if era == "Unknown":
            continue
        papers = data.get("papers", [])
        if not papers:
            continue

        era_citations = sum(p.get("cited_by_count", 0) for p in papers)
        era_avg = era_citations / len(papers) if papers else 0

        era_stats.append({
            "era": era,
            "paper_count": len(papers),
            "total_citations": era_citations,
            "avg_citations": era_avg,
            "milestone_count": data.get("milestone_count", 0),
            "start_year": data.get("start_year", 0),
        })
        total_citations += era_citations
        total_papers += len(papers)

    if not era_stats or total_papers == 0:
        return None

    overall_avg = total_citations / total_papers

    # Find era with highest citation ratio
    for stat in era_stats:
        stat["citation_ratio"] = stat["avg_citations"] / overall_avg if overall_avg > 0 else 0

    # Sort by citation ratio
    era_stats.sort(key=lambda s: -s["citation_ratio"])

    top_era = era_stats[0]

    # Threshold for breakthrough: ratio > 2.0 and at least 2 milestone papers
    if top_era["citation_ratio"] > 2.0 and top_era["milestone_count"] >= 2:
        return {
            "breakthrough_era": top_era["era"],
            "breakthrough_year": top_era["start_year"] + 5,  # Mid-decade estimate
            "citation_ratio": round(top_era["citation_ratio"], 2),
            "milestone_count": top_era["milestone_count"],
        }

    return None


def get_sample_abstracts_by_era(
    conn: Connection,
    era_data: Dict[str, Dict[str, Any]],
    before_era: str,
    after_era: str,
    max_per_era: int = 5,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Get sample paper abstracts from before and after a breakthrough era.

    Returns (before_papers, after_papers) with titles and abstracts.
    """
    before_papers = []
    after_papers = []

    # Get work_ids from each era
    before_data = era_data.get(before_era, {})
    after_data = era_data.get(after_era, {})

    before_work_ids = [p["work_id"] for p in before_data.get("papers", [])[:max_per_era]]
    after_work_ids = [p["work_id"] for p in after_data.get("papers", [])[:max_per_era]]

    all_work_ids = before_work_ids + after_work_ids

    if not all_work_ids:
        return [], []

    # Fetch titles and abstracts
    rows = conn.execute(
        text("""
            SELECT work_id, title, abstract
            FROM works
            WHERE work_id = ANY(:work_ids)
        """),
        {"work_ids": all_work_ids},
    ).mappings().all()

    paper_map = {row["work_id"]: row for row in rows}

    for wid in before_work_ids:
        if wid in paper_map:
            before_papers.append({
                "title": paper_map[wid]["title"],
                "abstract": _truncate_text(paper_map[wid]["abstract"] or "", 200),
            })

    for wid in after_work_ids:
        if wid in paper_map:
            after_papers.append({
                "title": paper_map[wid]["title"],
                "abstract": _truncate_text(paper_map[wid]["abstract"] or "", 200),
            })

    return before_papers, after_papers


def analyze_breakthrough_llm(
    before_papers: List[Dict[str, Any]],
    after_papers: List[Dict[str, Any]],
    breakthrough_year: int,
) -> Dict[str, Any]:
    """
    Use LLM to analyze methodology shift across a breakthrough.

    Returns dict with pre/post methodology descriptions and confidence.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, skipping breakthrough LLM analysis")
        return {
            "pre_breakthrough_approach": None,
            "post_breakthrough_approach": None,
            "confidence": 0.0,
        }

    # Build paper summaries
    before_text = "\n".join([
        f"- {p['title']}: {p['abstract']}" for p in before_papers
    ])
    after_text = "\n".join([
        f"- {p['title']}: {p['abstract']}" for p in after_papers
    ])

    prompt = f"""Analyze the methodology shift around year {breakthrough_year}.

PAPERS BEFORE {breakthrough_year}:
{before_text}

PAPERS AFTER {breakthrough_year}:
{after_text}

Identify:
1. The dominant methodology/approach BEFORE the transition
2. The dominant methodology/approach AFTER the transition
3. Your confidence (0-1) that this represents a true paradigm shift

Return JSON:
{{
    "pre_breakthrough_approach": "Methodology before (1 sentence)",
    "post_breakthrough_approach": "Methodology after (1 sentence)",
    "confidence": 0.8
}}"""

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": "You are a research methodology analyst. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                timeout=60.0,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            result = json.loads(content)
            return {
                "pre_breakthrough_approach": result.get("pre_breakthrough_approach"),
                "post_breakthrough_approach": result.get("post_breakthrough_approach"),
                "confidence": float(result.get("confidence", 0.5)),
            }

        except Exception as e:
            logger.warning(f"Breakthrough analysis LLM failed: {e}")
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                continue
            break

    return {
        "pre_breakthrough_approach": None,
        "post_breakthrough_approach": None,
        "confidence": 0.0,
    }


def analyze_era_evolution_llm(
    conn: Connection,
    era_data: Dict[str, Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Analyze methodology evolution across all eras using LLM.

    Returns tuple of (evolution_trends, paradigm_shift_info).
    - evolution_trends: List of EvolutionTrend dicts
    - paradigm_shift_info: Dict with breakthrough details if detected, else None
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, returning basic era info")
        return [], None

    # Get sample papers from each era
    era_samples = []
    sorted_eras = sorted(
        [e for e in era_data.keys() if e != "Unknown"],
        key=lambda e: era_data[e].get("start_year", 0)
    )

    for era in sorted_eras:
        data = era_data[era]
        papers = data.get("papers", [])[:5]  # Get top 5 papers per era for better grounding
        work_ids = [p["work_id"] for p in papers]

        if not work_ids:
            continue

        # Fetch titles AND abstracts for grounding
        rows = conn.execute(
            text("SELECT work_id, title, abstract FROM works WHERE work_id = ANY(:wids)"),
            {"wids": work_ids},
        ).mappings().all()

        # Build paper info with full abstracts (usually 200-500 words)
        paper_info = []
        for row in rows:
            if row["title"]:
                abstract = row["abstract"] or ""
                paper_info.append({
                    "title": row["title"],
                    "abstract": abstract if abstract else "(no abstract)",
                })

        era_samples.append({
            "era": era,
            "papers": paper_info,
            "paper_count": len(data.get("papers", [])),
        })

    if not era_samples:
        return [], None

    # Number all papers globally so LLM can reference [1], [2], etc.
    all_papers_numbered = []
    paper_idx = 1
    for es in era_samples:
        for p in es.get("papers", []):
            p["paper_number"] = paper_idx
            all_papers_numbered.append({"number": paper_idx, "title": p["title"]})
            paper_idx += 1
    total_papers = len(all_papers_numbered)

    prompt = f"""Analyze methodology evolution across eras. DEFAULT: no paradigm shift detected (most fields evolve incrementally).

PAPERS BY ERA (each paper has a "paper_number" - use [N] to cite):
{json.dumps(era_samples, indent=2)}
Total papers: {total_papers}

STEP 1 - EVOLUTION TRENDS (ALWAYS do this):
For EACH era, identify the dominant approach and key themes from the abstracts.

STEP 2 - VALIDATION CHECKLIST (answer BEFORE deciding detected):
Fill out ALL 5 fields honestly. DEFAULT is detected=false.

same_domain_same_problem: Are the OLD and NEW approaches solving the EXACT SAME technical problem?
  - "image classification" in both eras = true
  - "manufacturing quality control" vs "software testing" = false (DIFFERENT domains)
  - "statistical process control" vs "verification & validation" = false (DIFFERENT problems)
  - "traditional optimization" vs "AI optimization" for same problem = true
  - If papers across eras are from DIFFERENT subfields/conferences, likely false

is_reporting_or_guideline_change: Is the "shift" about HOW RESULTS ARE REPORTED, not how analysis is done?
  - PRISMA, CONSORT, ISO standards = true
  - New computational method = false

old_approach_still_works_for_same_tasks: Is the old method STILL WIDELY PRACTICED TODAY?
  - If yes → true → detected MUST be false
  - Supply chain optimization with LP/VRP = TRUE (still standard)
  - Manual systematic review = TRUE (still gold standard)
  - Traditional software testing = TRUE (still used everywhere)
  - SIFT for image classification = FALSE (nobody uses it anymore)
  - Rule-based NLP for text generation = FALSE (completely replaced)

new_capability_impossible_with_old: Can the old approach FUNDAMENTALLY NOT do what the new does?
  - "More accurate" or "faster" is NOT enough
  - SIFT cannot classify 1000 categories on ImageNet = true
  - Traditional optimization still solves supply chain problems = false

passes_all_criteria: Set true ONLY if: same_domain=true AND reporting=false AND old_works=false AND impossible=true

STEP 3 - DECISION:
- If passes_all_criteria=false → detected=false, write rejection_reason (6 sentences)
- If passes_all_criteria=true AND you are CERTAIN → detected=true, write shift_description (8 sentences)

Return JSON:
{{
    "evolution": [{{"era": "1990s", "dominant_approach": "...", "key_themes": ["...", "..."]}}],
    "paradigm_shift": {{
        "validation": {{
            "same_domain_same_problem": false,
            "is_reporting_or_guideline_change": false,
            "old_approach_still_works_for_same_tasks": true,
            "new_capability_impossible_with_old": false,
            "passes_all_criteria": false
        }},
        "detected": false,
        "magnitude": "minor",
        "from_era": "...",
        "to_era": "...",
        "breakthrough_year": null,
        "from_approach": "...",
        "to_approach": "...",
        "shift_description": null,
        "shift_citations": [],
        "rejection_reason": "6-sentence paragraph explaining why no paradigm shift",
        "rejection_citations": [],
        "confidence": 0.5
    }}
}}

NOTE: The JSON example above shows the DEFAULT state (no shift). Only change detected to true if validation passes.

REJECTION_REASON FORMAT (6 sentences - use when detected=false):
1. Name the SHARED core methodology across all eras
2. What the EARLIER era contributed (name specifically)
3. What the LATER era added (name specifically)
4. The CORE ASSUMPTION that persists
5. What a TRUE paradigm shift would require
6. Why that shift has NOT occurred

SHIFT_DESCRIPTION FORMAT (8 sentences - ONLY if detected=true):
Each sentence MUST cite a DIFFERENT paper [N]. You MUST cite at least 6 different paper_numbers.
1. Name the OLD technique BY NAME [N]
2. What the OLD technique COULD NOT do [N]
3. Name the NEW technique BY NAME [N]
4. The technical MECHANISM [N]
5. WHY better: "X WHICH causes Y and THEREFORE enables Z" [N]
6. SPECIFIC TASK demonstrated [N]
7. QUANTITATIVE comparison with relative improvement [N] vs [N]
8. How this changed research direction [N]

Example shift_description (~140 words, [1] through [7]):
"SIFT required manual specification of gradient histograms and keypoint detectors for each visual domain [1]. These handcrafted features could not generalize because edge detectors tuned for natural image statistics fail when applied to domains with different gradient distributions [2]. AlexNet learns hierarchical features directly from raw pixels using 60M parameters across 5 convolutional layers [3]. The mechanism is convolutional weight sharing: the same learned filters slide across the image, detecting patterns regardless of spatial position [4]. Training on 1.2 million ImageNet images [5], WHICH provides sufficient variation, THEREFORE allows the network to learn generalizable features without manual engineering. This enabled classification across 1000 object categories with a single architecture [5]. AlexNet achieved 15.3% top-5 error [3] versus 26.2% [6] for the previous best, a 42% relative reduction. This shifted research from crafting domain-specific features to designing deeper architectures [7]."

CITATION RULES (for shift_description):
- Each of the 8 sentences cites a DIFFERENT paper [N] - minimum 6 unique paper_numbers
- ALL claims grounded in PROVIDED abstracts only - no general knowledge
- shift_citations must list every claim with its source paper title

ADDITIONAL RULES:
- detected=true ONLY if magnitude=major AND confidence>=0.85 AND passes_all_criteria=true"""

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {"role": "system", "content": "You analyze methodology evolution in academic fields. DEFAULT OUTPUT: detected=false (most fields evolve incrementally, not through paradigm shifts).\n\nCRITICAL RULES:\n1. Fill out the 'validation' object FIRST. Default all booleans conservatively (same_domain=false, old_works=true, impossible=false).\n2. If ANY validation check fails → detected MUST be false. Write a rejection_reason instead.\n3. old_approach_still_works_for_same_tasks: If practitioners STILL USE the old method today (even if newer methods exist), set TRUE → detected=false. 'Slower' or 'less scalable' does NOT mean it stopped working.\n4. same_domain_same_problem: Papers must be solving the EXACT SAME technical problem. Different subfields/conferences = false.\n5. Only set detected=true if ALL 5 validation criteria pass AND magnitude=major AND confidence>=0.85.\n6. For shift_description: write 8 sentences, each citing a DIFFERENT paper [N] (minimum 6 unique). Ground ALL claims in provided abstracts.\n7. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                timeout=60.0,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            result = json.loads(content)
            logger.info(f"LLM paradigm_shift response: {result.get('paradigm_shift', {})}")

            # Extract evolution trends
            evolution = result.get("evolution", [])
            if isinstance(evolution, list):
                # Filter out non-dict entries and add representative paper
                evolution = [t for t in evolution if isinstance(t, dict)]
                for trend in evolution:
                    era = trend.get("era")
                    if era in era_data:
                        papers = era_data[era].get("papers", [])
                        if papers:
                            trend["representative_paper_id"] = papers[0]["work_id"]

            # Extract paradigm shift info
            paradigm_shift = None
            shift_data = result.get("paradigm_shift", {})
            if not isinstance(shift_data, dict):
                shift_data = {}
            magnitude = shift_data.get("magnitude", "minor")
            confidence = float(shift_data.get("confidence", 0.5))

            # Check validation field - REQUIRED for paradigm shift acceptance
            validation = shift_data.get("validation", {})
            passes_validation = validation.get("passes_all_criteria", False)  # Default false - must explicitly pass
            same_domain = validation.get("same_domain_same_problem", False)  # Default false - must prove same domain
            is_guideline_change = validation.get("is_reporting_or_guideline_change", True)  # Default true - assume guideline unless proven otherwise
            old_still_works = validation.get("old_approach_still_works_for_same_tasks", True)  # Default true - conservative
            new_impossible_old = validation.get("new_capability_impossible_with_old", False)  # Default false - must prove

            # Log validation results
            logger.info(
                f"Paradigm shift validation: passes={passes_validation}, "
                f"same_domain={same_domain}, "
                f"is_guideline={is_guideline_change}, "
                f"old_works={old_still_works}, "
                f"new_impossible_old={new_impossible_old}, "
                f"validation_obj={validation}"
            )

            # Only accept paradigm shifts that:
            # 1. Are major AND high confidence
            # 2. Pass validation (not a guideline/reporting change)
            # 3. Same domain/problem
            # 4. Validation field must be present and pass all criteria
            # Threshold of 0.85 ensures only clear paradigm shifts are reported
            # Code-level heuristic: reject shifts where the "new approach" is just
            # automating the same methodology (e.g., manual review → ML-assisted review)
            # This does NOT apply to genuine computational paradigm shifts like SIFT → CNNs
            from_approach = (shift_data.get("from_approach") or "").lower()
            to_approach = (shift_data.get("to_approach") or "").lower()
            shift_desc_lower = (shift_data.get("shift_description") or "").lower()
            is_automation_shift = (
                any(w in to_approach for w in ["machine learning assisted", "ml-assisted", "automated", "ai-assisted", "nlp-assisted"])
                and any(w in shift_desc_lower for w in ["manual screening", "manual extraction", "manual review", "hand-coded", "manual specification of study"])
            )
            if is_automation_shift:
                logger.info(
                    f"Paradigm shift REJECTED by heuristic: '{from_approach}' -> '{to_approach}' "
                    f"is an automation of the same methodology, not a paradigm shift"
                )

            is_valid_shift = (
                shift_data.get("detected")
                and magnitude == "major"
                and confidence >= 0.85
                and passes_validation
                and same_domain
                and not is_guideline_change
                and not old_still_works
                and new_impossible_old
                and not is_automation_shift
            )

            if is_valid_shift:
                to_era = shift_data.get("to_era")

                # Use LLM-provided breakthrough year, fallback to era mid-point
                breakthrough_year = shift_data.get("breakthrough_year")
                if not breakthrough_year and to_era and to_era in era_data:
                    breakthrough_year = era_data[to_era].get("start_year", 0) + 5

                # Get milestone papers from the breakthrough era (with titles)
                breakthrough_papers = []
                if to_era and to_era in era_data:
                    era_papers = era_data[to_era].get("papers", [])
                    # First try milestones
                    milestone_papers = [p for p in era_papers if p.get("is_milestone", False)]
                    # Then top by citation
                    source_papers = milestone_papers if milestone_papers else era_papers[:5]
                    breakthrough_papers = [
                        {
                            "work_id": p["work_id"],
                            "title": p.get("title"),
                            "cited_by_count": p.get("cited_by_count", 0),
                        }
                        for p in source_papers[:5]
                    ]

                # Validate citation diversity - retry once if too few unique papers
                shift_citations = shift_data.get("shift_citations", [])
                # Normalize: filter to dicts only (model sometimes returns strings)
                shift_citations = [c for c in shift_citations if isinstance(c, dict)]
                unique_papers_cited = len(set(
                    c.get("paper_title", "") for c in shift_citations if c.get("paper_title")
                ))
                if unique_papers_cited < 5 and total_papers >= 6:
                    logger.warning(
                        f"Shift description cites only {unique_papers_cited} unique papers, retrying with diversity instruction"
                    )
                    try:
                        retry_resp = client.chat.completions.create(
                            model=MODEL_VERSION,
                            messages=[
                                {"role": "system", "content": "Rewrite the shift_description and shift_citations. Each of the 8 sentences MUST cite a DIFFERENT paper [N]. You have these papers available:\n" + json.dumps(all_papers_numbered, indent=1) + "\nReturn JSON with ONLY shift_description (string) and shift_citations (list of {claim, paper_number, paper_title})."},
                                {"role": "user", "content": f"Current shift_description:\n{shift_data.get('shift_description')}\n\nRewrite so each sentence cites a DIFFERENT paper number. Minimum 6 unique paper_numbers out of 8 sentences."},
                            ],
                            timeout=60.0,
                            response_format={"type": "json_object"},
                        )
                        retry_content = (retry_resp.choices[0].message.content or "").strip()
                        if retry_content.startswith("```"):
                            rlines = retry_content.split("\n")
                            retry_content = "\n".join(rlines[1:-1] if rlines[-1].startswith("```") else rlines[1:])
                        retry_data = json.loads(retry_content)
                        new_citations = retry_data.get("shift_citations", [])
                        new_unique = len(set(c.get("paper_title", "") for c in new_citations if c.get("paper_title")))
                        if new_unique > unique_papers_cited:
                            shift_data["shift_description"] = retry_data.get("shift_description", shift_data.get("shift_description"))
                            shift_data["shift_citations"] = new_citations
                            shift_citations = new_citations
                            unique_papers_cited = new_unique
                            logger.info(f"Citation diversity retry improved to {new_unique} unique papers")
                    except Exception as retry_err:
                        logger.warning(f"Citation diversity retry failed: {retry_err}")

                paradigm_shift = {
                    "breakthrough_year": breakthrough_year,
                    "from_era": shift_data.get("from_era"),
                    "breakthrough_era": to_era,
                    "breakthrough_papers": breakthrough_papers,
                    "pre_breakthrough_approach": shift_data.get("from_approach"),
                    "post_breakthrough_approach": shift_data.get("to_approach"),
                    "shift_description": shift_data.get("shift_description"),
                    "shift_citations": shift_citations,
                    "unique_papers_cited": unique_papers_cited,
                    "confidence": confidence,
                }

                logger.info(
                    f"Paradigm shift accepted: {shift_data.get('from_approach')} -> "
                    f"{shift_data.get('to_approach')} (magnitude={magnitude}, conf={confidence:.2f})"
                )
            elif shift_data.get("detected"):
                # Paradigm shift detected but didn't meet threshold
                logger.info(
                    f"Paradigm shift REJECTED: magnitude={magnitude}, confidence={confidence:.2f} "
                    f"(need major + >=0.85)"
                )
                # Still return with rejection reason
                paradigm_shift = {
                    "rejection_reason": shift_data.get("rejection_reason") or f"Detected but below threshold: magnitude={magnitude}, confidence={confidence:.2f}. Requires major magnitude and confidence >= 0.85.",
                    "rejection_citations": shift_data.get("rejection_citations", []),
                    "confidence": confidence,
                }
            else:
                # No paradigm shift detected - return rejection reason
                rejection_reason = shift_data.get("rejection_reason")
                if rejection_reason:
                    paradigm_shift = {
                        "rejection_reason": rejection_reason,
                        "rejection_citations": shift_data.get("rejection_citations", []),
                        "confidence": confidence,
                    }

            return evolution, paradigm_shift

        except Exception as e:
            import traceback
            logger.warning(f"Evolution analysis LLM failed: {e}\n{traceback.format_exc()}")
            error_str = str(e).lower()
            is_transient = "rate" in error_str or "timeout" in error_str
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                continue
            break

    return [], None


def analyze_temporal_map(
    conn: Connection,
    papers: List[Dict[str, Any]],
    era_data: Dict[str, Dict[str, Any]],
    topic_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze a temporal map for breakthrough year and evolution trends.

    Uses LLM-based paradigm shift detection (more accurate for mature fields)
    with citation-based detection as fallback for emerging fields.

    Args:
        conn: Database connection
        papers: List of papers in the temporal map
        era_data: Dict of era -> era data
        topic_id: Optional topic ID for caching

    Returns:
        TemporalMapAnalytics dict
    """
    # Analyze evolution trends AND detect paradigm shifts via LLM
    evolution, llm_breakthrough = analyze_era_evolution_llm(conn, era_data)

    breakthrough_analysis = None

    # Primary: Use LLM-detected paradigm shift (works better for mature fields)
    if llm_breakthrough:
        breakthrough_analysis = llm_breakthrough
        logger.info(
            f"LLM detected paradigm shift: {llm_breakthrough.get('pre_breakthrough_approach')} -> "
            f"{llm_breakthrough.get('post_breakthrough_approach')} "
            f"(confidence: {llm_breakthrough.get('confidence', 0):.2f})"
        )

        # Mark breakthrough era in era_data
        breakthrough_era = llm_breakthrough.get("breakthrough_era")
        if breakthrough_era and breakthrough_era in era_data:
            era_data[breakthrough_era]["is_breakthrough_era"] = True

    # Fallback: Use citation-based detection (works for emerging fields with clear spikes)
    if not breakthrough_analysis:
        citation_breakthrough = detect_breakthrough_era(era_data)
        if citation_breakthrough:
            sorted_eras = sorted(
                [e for e in era_data.keys() if e != "Unknown"],
                key=lambda e: era_data[e].get("start_year", 0)
            )

            breakthrough_era = citation_breakthrough["breakthrough_era"]
            breakthrough_idx = sorted_eras.index(breakthrough_era) if breakthrough_era in sorted_eras else -1

            if breakthrough_idx > 0:
                before_era = sorted_eras[breakthrough_idx - 1]
                after_era = breakthrough_era

                # Get sample papers for LLM analysis
                before_papers, after_papers = get_sample_abstracts_by_era(
                    conn, era_data, before_era, after_era
                )

                # Analyze with LLM
                llm_result = analyze_breakthrough_llm(
                    before_papers,
                    after_papers,
                    citation_breakthrough["breakthrough_year"],
                )

                # Get breakthrough papers (milestones from breakthrough era)
                breakthrough_papers = [
                    p["work_id"]
                    for p in era_data.get(breakthrough_era, {}).get("papers", [])
                    if p.get("is_milestone", False)
                ][:5]

                breakthrough_analysis = {
                    "breakthrough_year": citation_breakthrough["breakthrough_year"],
                    "from_era": before_era,
                    "breakthrough_era": breakthrough_era,
                    "breakthrough_papers": breakthrough_papers,
                    "pre_breakthrough_approach": llm_result.get("pre_breakthrough_approach"),
                    "post_breakthrough_approach": llm_result.get("post_breakthrough_approach"),
                    "shift_description": None,  # Citation-based detection doesn't generate description
                    "confidence": llm_result.get("confidence", 0.0),
                }

                # Mark breakthrough era in era_data
                if breakthrough_era in era_data:
                    era_data[breakthrough_era]["is_breakthrough_era"] = True

                logger.info(
                    f"Citation-based breakthrough detected: {breakthrough_era} "
                    f"(ratio: {citation_breakthrough.get('citation_ratio', 0):.2f})"
                )

    return {
        "breakthrough": breakthrough_analysis,
        "evolution": evolution,
    }
