"""
Methodology comparison service for Feature 4.

Orchestrates the full pipeline:
1. Validate inputs (work_ids belong to map)
2. Check comparison cache
3. Fetch paper content (S2 full text / abstract) — Stage 1A
4. Build citation lineage — Stage 1B
5. Extract methodology fingerprints (LLM call 1) — Stage 2
6. Synthesize comparison (LLM call 2) — Stage 3
7. Cache and return
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection, Engine

from app.feature3.json_utils import extract_json_from_llm_response
from app.feature4.citation_lineage import build_citation_lineage
from app.feature4.paper_content import PaperContent, fetch_papers_content
from app.feature4.schemas import (
    Assumption,
    Capability,
    CitationLineage,
    CommonProblem,
    Complement,
    ConvergenceDivergence,
    DecisionScenario,
    Limitation,
    MethodologyComparisonResponse,
    MethodologyFingerprint,
    PaperMethodProfile,
    PaperStrengthsWeaknesses,
    Paradigm,
    Recommendation,
)

# Load env
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 0.5


def _get_client() -> Optional[OpenAI]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found")
        return None
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _ensure_str(val: Any) -> str:
    """Coerce LLM output to string (handles list, dict, etc.)."""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


def _comparison_hash(work_ids: list[str]) -> str:
    """Deterministic hash for a set of work_ids."""
    key = "|".join(sorted(work_ids))
    return hashlib.sha256(key.encode()).hexdigest()


def _get_cached_comparison(
    conn: Connection, work_ids: list[str]
) -> Optional[Dict[str, Any]]:
    h = _comparison_hash(work_ids)
    row = conn.execute(
        sa_text("""
            SELECT result_json FROM methodology_comparison_cache
            WHERE comparison_hash = :h
        """),
        {"h": h},
    ).mappings().first()
    if row:
        val = row["result_json"]
        return val if isinstance(val, dict) else json.loads(val)
    return None


def _cache_comparison(
    conn: Connection,
    tenant_id: UUID,
    work_ids: list[str],
    result: Dict[str, Any],
) -> None:
    try:
        conn.execute(
            sa_text("""
                INSERT INTO methodology_comparison_cache
                    (comparison_hash, tenant_id, work_ids, result_json, model_version)
                VALUES (:h, :tid, :wids, :rj, :mv)
                ON CONFLICT (comparison_hash) DO UPDATE SET
                    result_json = EXCLUDED.result_json,
                    model_version = EXCLUDED.model_version,
                    created_at = now()
            """),
            {
                "h": _comparison_hash(work_ids),
                "tid": str(tenant_id),
                "wids": work_ids,
                "rj": json.dumps(result),
                "mv": MODEL_VERSION,
            },
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to cache comparison: {e}")


def _get_cached_fingerprint(
    conn: Connection, work_id: str
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        sa_text("""
            SELECT fingerprint_json, source_quality
            FROM methodology_fingerprint_cache
            WHERE work_id = :wid
        """),
        {"wid": work_id},
    ).mappings().first()
    if row:
        fp = row["fingerprint_json"]
        return {
            "fingerprint": fp if isinstance(fp, dict) else json.loads(fp),
            "source_quality": row["source_quality"],
        }
    return None


def _cache_fingerprint(
    conn: Connection,
    work_id: str,
    fingerprint: Dict[str, Any],
    source_quality: str,
) -> None:
    try:
        conn.execute(
            sa_text("""
                INSERT INTO methodology_fingerprint_cache
                    (work_id, fingerprint_json, source_quality, model_version)
                VALUES (:wid, :fp, :sq, :mv)
                ON CONFLICT (work_id) DO UPDATE SET
                    fingerprint_json = EXCLUDED.fingerprint_json,
                    source_quality = EXCLUDED.source_quality,
                    model_version = EXCLUDED.model_version,
                    created_at = now()
            """),
            {
                "wid": work_id,
                "fp": json.dumps(fingerprint),
                "sq": source_quality,
                "mv": MODEL_VERSION,
            },
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to cache fingerprint for {work_id}: {e}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_work_ids_in_map(
    conn: Connection, map_id: str, work_ids: list[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Verify all work_ids belong to the map. Returns metadata dict keyed by work_id.
    Raises ValueError if any work_id is missing.
    """
    rows = conn.execute(
        sa_text("""
            SELECT mn.work_id, w.title, w.year, w.venue, w.cited_by_count,
                   w.doi, w.arxiv_id, w.abstract, w.referenced_works_json
            FROM map_nodes mn
            JOIN works w ON w.work_id = mn.work_id
            WHERE mn.map_id = :map_id
              AND mn.work_id = ANY(:ids)
        """),
        {"map_id": map_id, "ids": work_ids},
    ).mappings().all()

    found = {row["work_id"]: dict(row) for row in rows}
    missing = set(work_ids) - set(found.keys())
    if missing:
        raise ValueError(
            f"work_ids not found in map {map_id}: {sorted(missing)}"
        )
    return found


def _validate_work_ids_in_rank_results(
    conn: Connection, rank_job_id: str, work_ids: list[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Verify all work_ids belong to a rank job's results. Returns metadata dict keyed by work_id.
    Raises ValueError if any work_id is missing.
    """
    rows = conn.execute(
        sa_text("""
            SELECT rr.work_id, w.title, w.year, w.venue, w.cited_by_count,
                   w.doi, w.arxiv_id, w.abstract, w.referenced_works_json
            FROM rank_results rr
            JOIN works w ON w.work_id = rr.work_id
            WHERE rr.rank_job_id = :rank_job_id
              AND rr.work_id = ANY(:ids)
        """),
        {"rank_job_id": rank_job_id, "ids": work_ids},
    ).mappings().all()

    found = {row["work_id"]: dict(row) for row in rows}
    missing = set(work_ids) - set(found.keys())
    if missing:
        raise ValueError(
            f"work_ids not found in rank job {rank_job_id}: {sorted(missing)}"
        )
    return found


# ---------------------------------------------------------------------------
# LLM Call 1: Methodology Extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "You are an expert research methodology analyst. "
    "You extract detailed technical profiles from academic papers. "
    "Follow the user's format instructions exactly. Return valid JSON only."
)


def _build_extraction_prompt(
    papers: list[Dict[str, Any]],
    contents: list[PaperContent],
) -> str:
    """Build the extraction prompt for uncached papers."""
    content_map = {c.work_id: c for c in contents}
    sections = []

    for i, paper in enumerate(papers, 1):
        wid = paper["work_id"]
        content = content_map.get(wid)
        text = content.text if content else (paper.get("abstract") or "")
        source = content.source_quality if content else "abstract_only"

        sections.append(
            f"--- PAPER {i} ---\n"
            f"Work ID: {wid}\n"
            f"Title: {paper.get('title', 'Unknown')}\n"
            f"Year: {paper.get('year', 'Unknown')}\n"
            f"Venue: {paper.get('venue', 'Unknown')}\n"
            f"Cited by: {paper.get('cited_by_count', 0)} papers\n"
            f"Source: {source}\n\n"
            f"{text}\n"
            f"--- END PAPER {i} ---"
        )

    return (
        "Analyze the following paper(s) and extract detailed methodology profiles.\n\n"
        "=== RULES ===\n"
        "1. 'approach': ALWAYS write 4-6 sentences covering: (1) the core architecture, "
        "(2) the loss/objective formulation with named terms, (3) training procedure with "
        "optimizer and hyperparameters. For abstract_only sources, supplement with your "
        "domain knowledge — write the SAME level of detail as for full_text.\n"
        "2. 'key_components': list NAMED techniques only (e.g., 'bottleneck block (1×1→3×3→1×1)'), "
        "not generic terms like 'deep neural networks' or 'residual connections'.\n"
        "3. 'assumptions': list 2-4 SPECIFIC technical assumptions the method makes. "
        "Examples: 'Input images are spatially aligned', 'Features at different depths have "
        "complementary information', 'Objects can be localized with rectangular bounding boxes', "
        "'Channel relationships capture semantic importance'. NOT generic like 'data is IID'.\n"
        "4. 'limitations': each MUST use the format '[design choice] → [consequence]'.\n"
        "5. All claims must be definitive. Do not use 'may', 'might', 'could', 'potentially', "
        "'careful tuning', or any qualifying language.\n\n"
        + "\n\n".join(sections)
        + "\n\n=== EXAMPLES ===\n"
        "GOOD extraction (full_text):\n"
        "approach: \"Proposes a conditional GAN with a U-Net generator and PatchGAN "
        "discriminator. Trained with λ·L1(G(x),y) + L_cGAN(G,D) where λ=100. "
        "The PatchGAN classifies 70×70 patches. Uses Adam (lr=0.0002, β1=0.5) for 200 epochs.\"\n"
        "key_components: [\"U-Net generator with skip connections\", "
        "\"PatchGAN 70×70 discriminator\", \"L1 loss (λ=100)\", \"cGAN adversarial loss\"]\n"
        "assumptions: [\"Input and output images are spatially aligned at pixel level\", "
        "\"Local patch statistics capture perceptual quality\", "
        "\"L1 loss encourages low-frequency correctness while adversarial loss captures high-frequency detail\"]\n"
        "limitations: [\"Requires pixel-aligned paired data, unlike CycleGAN\", "
        "\"Mode collapse causes repeated output textures\"]\n\n"
        "EXAMPLE of a GOOD extraction (abstract_only — supplement with domain knowledge):\n"
        "approach: \"Introduces cycle-consistent adversarial networks for unpaired image "
        "translation. Learns two generators G: X→Y and F: Y→X with cycle consistency "
        "loss F(G(X))≈X and G(F(Y))≈Y. Uses ResNet-9 generators with 9 residual blocks "
        "and PatchGAN discriminators. Trained with Adam (lr=0.0002, β1=0.5) using "
        "least-squares GAN loss plus cycle consistency loss (λ=10).\"\n"
        "key_components: [\"ResNet-9 generator\", \"PatchGAN discriminator\", "
        "\"cycle consistency loss (λ=10)\", \"least-squares adversarial loss\", "
        "\"identity loss for color preservation\"]\n"
        "assumptions: [\"Bijective mapping exists between source and target domains\", "
        "\"Cycle consistency is a sufficient proxy for semantic correspondence\", "
        "\"Domain-specific styles are separable from content\"]\n"
        "limitations: [\"Bijective mapping assumption → fails for many-to-one translations "
        "where multiple source images map to one target\", \"Two generators + two discriminators "
        "→ 2× the parameters and training time of single-direction methods\"]\n\n"
        "Return JSON:\n"
        "{\n"
        '  "papers": [\n'
        "    {\n"
        '      "work_id": "...",\n'
        '      "approach": "Detailed description of the core method (3-5 sentences). Include '
        'the key technical innovation, how it works mechanistically, and what problem formulation '
        'it uses (e.g., optimization objective, loss function, statistical framework).",\n'
        '      "data_requirements": "Specific data types, formats, and scale needed '
        '(e.g., paired vs unpaired images, labeled vs unlabeled, dataset sizes used)",\n'
        '      "assumptions": ["Specific technical assumption 1", "Assumption 2", "Assumption 3 (2-4 required)"],\n'
        '      "validation_method": "Exact metrics, benchmarks, baselines, and datasets used '
        'for evaluation",\n'
        '      "limitations": ["[design choice] → [concrete consequence]", ...],\n'
        '      "domain": "Problem domain and subfield",\n'
        '      "key_components": ["Exact named technique/algorithm/architecture 1", ...],\n'
        '      "novelty_over_prior": "Name the prior method(s) this builds on and what '
        'specific change was made (null if not stated)"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


import re as _re

_HEDGING_PATTERN = _re.compile(
    r'\b(may\s|might\s|could\s|potentially|possibly|careful tuning)',
    _re.IGNORECASE,
)


def _validate_extraction(result: Dict[str, Any]) -> list[str]:
    """Validate extraction output. Returns list of violation descriptions."""
    violations = []
    for paper in result.get("papers", []):
        wid = paper.get("work_id", "?")
        approach = paper.get("approach", "")
        if len(approach) < 200:
            violations.append(
                f"{wid} approach is too short ({len(approach)} chars). "
                "Write 4-6 detailed sentences with architecture, loss function, "
                "and training procedure."
            )
        # Validate assumptions (require 2-4)
        assumptions = paper.get("assumptions", [])
        if len(assumptions) < 2:
            violations.append(
                f"{wid} has only {len(assumptions)} assumption(s). "
                "Provide 2-4 specific technical assumptions (e.g., 'Input images are spatially aligned', "
                "'Channel relationships capture semantic importance')."
            )
        for comp in paper.get("key_components", []):
            lower = comp.lower()
            if lower in ("deep neural networks", "residual connections",
                         "convolutional networks", "deep learning"):
                violations.append(
                    f"{wid} key_component '{comp}' is too generic. "
                    "Use specific named techniques (e.g., 'bottleneck block (1×1→3×3→1×1)')."
                )
        for lim in paper.get("limitations", []):
            if _HEDGING_PATTERN.search(lim):
                violations.append(
                    f"{wid} limitation contains hedging: '{lim[:80]}...'. "
                    "Rewrite as '[design choice] → [consequence]' with no qualifying words."
                )
            if "→" not in lim and " -> " not in lim:
                violations.append(
                    f"{wid} limitation missing '→' structure: '{lim[:80]}...'. "
                    "Format: '[design choice] → [consequence]'."
                )
    return violations


_OVERLAP_STOP = {"and", "the", "of", "in", "for", "a", "an", "vs", "with", "to",
                  "on", "by", "is", "are", "it", "its", "this", "that", "each",
                  "both", "using", "uses", "more", "less", "over", "than"}

_EQUIV_CLUSTERS = [
    {"connectivity", "feature", "reuse", "mechanism", "flow", "gradient"},
    {"architecture", "network", "extraction", "design", "structure"},
    {"task", "scope", "capability", "focus", "orientation"},
    {"representational", "power", "capacity", "expressiveness"},
]


def _dim_keywords(s: str) -> set[str]:
    return {w.lower() for w in _re.findall(r'[a-z]+', s.lower())} - _OVERLAP_STOP


def _in_same_cluster(kw_a: set[str], kw_b: set[str]) -> bool:
    for cluster in _EQUIV_CLUSTERS:
        if (kw_a & cluster) and (kw_b & cluster):
            if len((kw_a & cluster) | (kw_b & cluster)) >= 2:
                return True
    return False


def _dimensions_overlap(dim_a: str, dim_b: str, texts_a: list[str], texts_b: list[str]) -> bool:
    """Detect if two dimensions are semantically overlapping."""
    kw_a, kw_b = _dim_keywords(dim_a), _dim_keywords(dim_b)

    # Name jaccard
    if kw_a and kw_b:
        if len(kw_a & kw_b) / len(kw_a | kw_b) >= 0.4:
            return True

    # Semantic cluster
    if kw_a and kw_b and _in_same_cluster(kw_a, kw_b):
        return True

    # Content overlap
    body_a = " ".join(texts_a)
    body_b = " ".join(texts_b)
    cw_a, cw_b = _dim_keywords(body_a), _dim_keywords(body_b)
    if cw_a and cw_b:
        if len(cw_a & cw_b) / len(cw_a | cw_b) >= 0.5:
            return True

    return False


def _is_shallow_cross_task(diffs: list[Dict[str, Any]]) -> bool:
    """Detect if a comparison is just 'does X / doesn't do X' repeated.

    Heuristic: if >50% of per_paper values contain negation patterns like
    'does not', 'limited to', 'not designed for', 'not directly', 'lacks',
    it's a shallow cross-task restatement.
    """
    _NEGATION = _re.compile(
        r'(does not|do not|limited to|not designed|not directly|lacks the|'
        r'cannot |is not |not applicable|not suitable)',
        _re.IGNORECASE,
    )
    total = 0
    negated = 0
    for diff in diffs:
        for text in diff.get("per_paper", {}).values():
            total += 1
            if _NEGATION.search(text):
                negated += 1
    if total == 0:
        return False
    return negated / total > 0.5


def _validate_synthesis(result: Dict[str, Any], work_ids: list[str]) -> list[str]:
    """Validate synthesis output. Returns list of violation descriptions."""
    violations = []

    # Validate convergence_divergence section
    conv_div = result.get("convergence_divergence", {})
    if not conv_div:
        violations.append("Missing 'convergence_divergence' section.")
    else:
        # Validate common_problem
        common_prob = conv_div.get("common_problem", {})
        if not common_prob.get("domain"):
            violations.append("convergence_divergence.common_problem.domain is missing or empty.")
        if not common_prob.get("challenge"):
            violations.append("convergence_divergence.common_problem.challenge is missing or empty.")
        if not common_prob.get("why_hard"):
            violations.append("convergence_divergence.common_problem.why_hard is missing or empty.")

        # Validate paradigms
        paradigms = conv_div.get("paradigms", [])
        if len(paradigms) < 1:
            violations.append("convergence_divergence.paradigms must have at least 1 paradigm.")
        for i, p in enumerate(paradigms):
            if not p.get("name"):
                violations.append(f"Paradigm {i+1} is missing 'name'.")
            if not p.get("papers") or not isinstance(p.get("papers"), list):
                violations.append(f"Paradigm {i+1} '{p.get('name', '?')}' is missing 'papers' list.")
            if not p.get("mechanism"):
                violations.append(f"Paradigm {i+1} '{p.get('name', '?')}' is missing 'mechanism'.")
            if not p.get("philosophy"):
                violations.append(f"Paradigm {i+1} '{p.get('name', '?')}' is missing 'philosophy'.")

        if not conv_div.get("divergence_summary"):
            violations.append("convergence_divergence.divergence_summary is missing or empty.")

    # Validate strengths_weaknesses_matrix section
    sw_matrix = result.get("strengths_weaknesses_matrix", [])
    if not sw_matrix:
        violations.append("Missing 'strengths_weaknesses_matrix' section.")
    else:
        found_wids = set()
        for sw in sw_matrix:
            wid = sw.get("work_id", "")
            found_wids.add(wid)

            # Check handles_well
            handles = sw.get("handles_well", [])
            if len(handles) < 1:
                violations.append(f"{wid} must have at least 1 'handles_well' entry.")
            for h in handles:
                if not h.get("capability"):
                    violations.append(f"{wid} handles_well entry missing 'capability'.")
                if not h.get("mechanism"):
                    violations.append(f"{wid} handles_well entry missing 'mechanism'.")
                if not h.get("evidence"):
                    violations.append(f"{wid} handles_well entry missing 'evidence'.")

            # Check struggles_with
            struggles = sw.get("struggles_with", [])
            if len(struggles) < 1:
                violations.append(f"{wid} must have at least 1 'struggles_with' entry.")
            for s in struggles:
                if not s.get("limitation"):
                    violations.append(f"{wid} struggles_with entry missing 'limitation'.")
                if not s.get("cause"):
                    violations.append(f"{wid} struggles_with entry missing 'cause'.")
                if not s.get("consequence"):
                    violations.append(f"{wid} struggles_with entry missing 'consequence'.")

            # Check assumptions
            assumptions = sw.get("assumptions", [])
            if len(assumptions) < 1:
                violations.append(f"{wid} must have at least 1 'assumptions' entry.")
            for a in assumptions:
                if not a.get("assumption"):
                    violations.append(f"{wid} assumptions entry missing 'assumption'.")
                if not a.get("if_violated"):
                    violations.append(f"{wid} assumptions entry missing 'if_violated'.")

            # Check complemented_by (cross-references)
            complements = sw.get("complemented_by", [])
            for c in complements:
                if not c.get("other_work_id"):
                    violations.append(f"{wid} complemented_by entry missing 'other_work_id'.")
                if not c.get("coverage"):
                    violations.append(f"{wid} complemented_by entry missing 'coverage'.")

        # Check all work_ids are covered
        missing = set(work_ids) - found_wids
        if missing:
            violations.append(f"strengths_weaknesses_matrix missing entries for: {sorted(missing)}")

    # Validate recommendation section
    rec = result.get("recommendation", {})
    if not rec:
        violations.append("Missing 'recommendation' section.")
    else:
        if not rec.get("summary"):
            violations.append("recommendation.summary is missing or empty.")

        decision_matrix = rec.get("decision_matrix", [])
        if len(decision_matrix) < 3:
            violations.append(
                f"recommendation.decision_matrix must have at least 3 scenarios "
                f"(found {len(decision_matrix)}). Cover different research contexts."
            )
        for i, d in enumerate(decision_matrix):
            if not d.get("scenario"):
                violations.append(f"Decision scenario {i+1} missing 'scenario'.")
            use_wid = d.get("use")
            if not use_wid:
                violations.append(f"Decision scenario {i+1} missing 'use' (work_id).")
            elif use_wid not in work_ids:
                violations.append(
                    f"Decision scenario {i+1} 'use' value '{use_wid}' is not a valid work_id. "
                    f"Must be one of: {work_ids}"
                )
            if not d.get("why"):
                violations.append(f"Decision scenario {i+1} missing 'why'.")

        if "can_combine" not in rec:
            violations.append("recommendation.can_combine is missing (must be true or false).")

    # Check for hedging in entire output
    text = json.dumps(result)
    hedges = _HEDGING_PATTERN.findall(text)
    if hedges:
        violations.append(
            f"Hedging language found: {hedges[:5]}. "
            "Rewrite all statements as definitive claims without 'may', 'might', 'could', 'potentially'."
        )

    return violations


def _call_llm(
    system: str,
    user: str,
    validator: Optional[Any] = None,
    validator_args: tuple = (),
) -> Optional[Dict[str, Any]]:
    """Make an LLM call with retries and optional validation.

    If validator is provided, it's called as validator(result, *validator_args).
    If it returns violations, the LLM is re-prompted with corrective feedback.
    """
    client = _get_client()
    if not client:
        return None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=messages,
                temperature=0.2,
                timeout=90.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            result, error = extract_json_from_llm_response(
                content, expected_type="object"
            )
            if result is None:
                logger.warning(f"LLM JSON parse failed (attempt {attempt+1}): {error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
                return None

            # Run validator if provided
            if validator is not None:
                violations = validator(result, *validator_args)
                if violations and attempt < MAX_RETRIES - 1:
                    logger.info(
                        f"Validation failed (attempt {attempt+1}): "
                        f"{len(violations)} violations"
                    )
                    # Append assistant response and corrective feedback
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your output has format violations. Fix ALL of them and "
                            "return the corrected JSON:\n\n"
                            + "\n".join(f"- {v}" for v in violations[:10])
                        ),
                    })
                    time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue

            return result
        except Exception as e:
            logger.warning(f"LLM call exception (attempt {attempt+1}): {e}")
            err = str(e).lower()
            is_transient = "rate" in err or "timeout" in err or "connection" in err
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            return None

    return None


def _extract_fingerprints(
    conn: Connection,
    papers_meta: list[Dict[str, Any]],
    contents: list[PaperContent],
) -> Dict[str, Dict[str, Any]]:
    """
    Extract methodology fingerprints. Uses cache where available,
    calls LLM for uncached papers in a single batch.
    """
    content_map = {c.work_id: c for c in contents}
    fingerprints: Dict[str, Dict[str, Any]] = {}
    uncached_papers: list[Dict[str, Any]] = []

    # Check cache
    for paper in papers_meta:
        wid = paper["work_id"]
        cached = _get_cached_fingerprint(conn, wid)
        if cached:
            logger.info(f"Fingerprint cache hit for {wid}")
            fingerprints[wid] = cached["fingerprint"]
        else:
            uncached_papers.append(paper)

    if not uncached_papers:
        return fingerprints

    # LLM call for uncached papers
    logger.info(f"Extracting fingerprints for {len(uncached_papers)} papers via LLM")
    uncached_contents = [
        content_map[p["work_id"]]
        for p in uncached_papers
        if p["work_id"] in content_map
    ]
    prompt = _build_extraction_prompt(uncached_papers, uncached_contents)
    result = _call_llm(
        EXTRACTION_SYSTEM, prompt,
        validator=_validate_extraction,
    )

    if result and "papers" in result:
        for paper_fp in result["papers"]:
            wid = paper_fp.get("work_id")
            if not wid or wid not in {p["work_id"] for p in uncached_papers}:
                continue
            fp = {
                "approach": paper_fp.get("approach", "Unknown"),
                "data_requirements": paper_fp.get("data_requirements", "Unknown"),
                "assumptions": paper_fp.get("assumptions", []),
                "validation_method": paper_fp.get("validation_method", "Unknown"),
                "limitations": paper_fp.get("limitations", []),
                "domain": paper_fp.get("domain", "Unknown"),
                "key_components": paper_fp.get("key_components", []),
                "novelty_over_prior": paper_fp.get("novelty_over_prior"),
            }
            fingerprints[wid] = fp
            sq = content_map[wid].source_quality if wid in content_map else "abstract_only"
            _cache_fingerprint(conn, wid, fp, sq)
    else:
        logger.error("LLM extraction returned no valid papers")
        # Fill defaults for uncached
        for paper in uncached_papers:
            wid = paper["work_id"]
            fingerprints[wid] = {
                "approach": "Could not extract methodology",
                "data_requirements": "Unknown",
                "assumptions": [],
                "validation_method": "Unknown",
                "limitations": [],
                "domain": "Unknown",
                "key_components": [],
                "novelty_over_prior": None,
            }

    return fingerprints


# ---------------------------------------------------------------------------
# LLM Call 2: Comparative Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = (
    "You are an expert research methodology analyst. "
    "You produce structured JSON comparisons of academic paper methodologies. "
    "Follow the user's format instructions exactly."
)


def _build_synthesis_prompt(
    papers_meta: list[Dict[str, Any]],
    fingerprints: Dict[str, Dict[str, Any]],
    contents: list[PaperContent],
    lineage: CitationLineage,
) -> str:
    content_map = {c.work_id: c for c in contents}

    # Paper profiles section
    profiles = []
    for paper in papers_meta:
        wid = paper["work_id"]
        sq = content_map[wid].source_quality if wid in content_map else "abstract_only"
        fp = fingerprints.get(wid, {})
        profiles.append(
            f"Paper ({wid}):\n"
            f"  Title: {paper.get('title', 'Unknown')} | "
            f"Year: {paper.get('year', 'Unknown')} | "
            f"Source: {sq}\n"
            f"  Profile: {json.dumps(fp, indent=2)}"
        )

    # Citation relationships section
    citation_lines = []
    for dc in lineage.direct_citations:
        citation_lines.append(f"- {dc.from_work_id} cites {dc.to_work_id}")

    for sr in lineage.shared_references:
        citers = ", ".join(sr.cited_by)
        citation_lines.append(
            f"- Shared reference: \"{sr.title}\" — cited by {citers}"
        )

    if lineage.evolution_chain:
        citation_lines.append(f"- Evolution: {lineage.evolution_chain}")

    citations_text = "\n".join(citation_lines) if citation_lines else "No direct citation relationships found between selected papers."

    work_id_list = [p["work_id"] for p in papers_meta]
    titles_map = {p["work_id"]: p.get("title", "Unknown") for p in papers_meta}

    return (
        "=== METHODOLOGY PROFILES ===\n\n"
        + "\n\n".join(profiles)
        + "\n\n=== CITATION RELATIONSHIPS ===\n"
        + citations_text
        + "\n\n=== YOUR TASK ===\n"
        "You are helping researchers decide which paper(s) to use for their work. "
        "Produce a structured comparison with THREE sections:\n\n"
        "1. CONVERGENCE/DIVERGENCE ANALYSIS (paradigm-centered)\n"
        "   - Identify the COMMON PROBLEM all papers address\n"
        "   - Group papers by their PARADIGM (methodological approach)\n"
        "   - Explain how each paradigm tackles the problem differently\n\n"
        "2. STRENGTHS/WEAKNESSES MATRIX (paper-centered)\n"
        "   - For EACH paper: what it handles well, what it struggles with, key assumptions\n"
        "   - Cross-reference: which other paper covers this paper's weakness?\n\n"
        "3. RECOMMENDATION (decision-focused)\n"
        "   - Your quick take on the landscape\n"
        "   - Decision matrix: specific scenarios → which paper → why\n"
        "   - Can these methods be combined? How?\n\n"
        "=== FORMAT RULES ===\n"
        "1. All claims must be DEFINITIVE. No 'may', 'might', 'could', 'potentially'.\n"
        "2. Be SPECIFIC with technical mechanisms, not vague descriptions.\n"
        "3. Every limitation must explain the CAUSE (design choice) and CONSEQUENCE.\n"
        "4. For abstract_only papers, use domain knowledge to fill in details.\n"
        "5. Provide at least 3 decision scenarios covering different research contexts.\n\n"
        "=== EXAMPLE OUTPUT ===\n"
        '{\n'
        '  "convergence_divergence": {\n'
        '    "common_problem": {\n'
        '      "domain": "Computer Vision",\n'
        '      "challenge": "Training very deep neural networks without gradient degradation",\n'
        '      "why_hard": "Gradients vanish/explode as network depth increases beyond ~20 layers, '
        'preventing effective learning in deeper architectures"\n'
        '    },\n'
        '    "paradigms": [\n'
        '      {\n'
        '        "name": "Additive Residual Learning",\n'
        '        "papers": ["Wpaper1"],\n'
        '        "mechanism": "Learns residual functions F(x) and adds them to identity shortcuts: y = F(x) + x",\n'
        '        "philosophy": "Easier to learn small perturbations than full mappings; identity is a reasonable default"\n'
        '      },\n'
        '      {\n'
        '        "name": "Dense Feature Aggregation",\n'
        '        "papers": ["Wpaper2"],\n'
        '        "mechanism": "Concatenates all preceding feature maps as input to each layer",\n'
        '        "philosophy": "Maximum feature reuse through direct connections; every layer should access all prior information"\n'
        '      }\n'
        '    ],\n'
        '    "divergence_summary": "Both papers solve gradient flow but trade off memory for feature access: '
        'ResNet uses constant-memory identity shortcuts while DenseNet uses linear-memory concatenation for full feature reuse."\n'
        '  },\n'
        '  "strengths_weaknesses_matrix": [\n'
        '    {\n'
        '      "work_id": "Wpaper1",\n'
        '      "title": "Deep Residual Learning",\n'
        '      "handles_well": [\n'
        '        {\n'
        '          "capability": "Training networks with 100+ layers",\n'
        '          "mechanism": "Identity shortcuts provide gradient highway regardless of depth",\n'
        '          "evidence": "Successfully trained 152-layer network on ImageNet; 1000-layer on CIFAR"\n'
        '        },\n'
        '        {\n'
        '          "capability": "Memory-efficient deep training",\n'
        '          "mechanism": "Additive residuals have O(1) memory overhead per block",\n'
        '          "evidence": "152-layer ResNet fits in 12GB GPU memory"\n'
        '        }\n'
        '      ],\n'
        '      "struggles_with": [\n'
        '        {\n'
        '          "limitation": "Feature reuse across distant layers",\n'
        '          "cause": "Each layer only accesses previous layer output + skip",\n'
        '          "consequence": "Features from early layers are not directly available to later layers"\n'
        '        }\n'
        '      ],\n'
        '      "assumptions": [\n'
        '        {\n'
        '          "assumption": "Residual functions are easier to optimize than full mappings",\n'
        '          "if_violated": "Network degrades to learning identity, gaining no representational power"\n'
        '        }\n'
        '      ],\n'
        '      "complemented_by": [\n'
        '        {\n'
        '          "other_work_id": "Wpaper2",\n'
        '          "coverage": "DenseNet provides direct feature reuse across all layers through concatenation"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ],\n'
        '  "recommendation": {\n'
        '    "summary": "ResNet excels for memory-constrained deep training; DenseNet for parameter-efficient models. '
        'Choose based on your GPU memory budget and whether model size or training memory matters more.",\n'
        '    "decision_matrix": [\n'
        '      {\n'
        '        "scenario": "Training very deep networks (100+ layers) on limited GPU memory",\n'
        '        "use": "Wpaper1",\n'
        '        "why": "Constant memory overhead per block allows scaling depth without memory explosion"\n'
        '      },\n'
        '      {\n'
        '        "scenario": "Deploying to edge devices where model size matters",\n'
        '        "use": "Wpaper2",\n'
        '        "why": "Dense connections achieve equivalent accuracy with 3x fewer parameters"\n'
        '      },\n'
        '      {\n'
        '        "scenario": "Transfer learning with limited target domain data",\n'
        '        "use": "Wpaper2",\n'
        '        "why": "Feature reuse across layers provides richer representations for fine-tuning"\n'
        '      }\n'
        '    ],\n'
        '    "can_combine": true,\n'
        '    "combination_notes": "ResNet-style shortcuts can be added within DenseNet blocks (DenseNet-BC). '
        'Alternatively, use ResNet backbone with DenseNet-style connections in specific modules."\n'
        '  }\n'
        '}\n\n'
        "=== NOW PRODUCE YOUR COMPARISON ===\n"
        "Analyze the papers above. Return JSON with this EXACT structure:\n"
        "{\n"
        '  "convergence_divergence": {\n'
        '    "common_problem": {"domain": "...", "challenge": "...", "why_hard": "..."},\n'
        '    "paradigms": [{"name": "...", "papers": ' + json.dumps(work_id_list[:1]) + ', "mechanism": "...", "philosophy": "..."}, ...],\n'
        '    "divergence_summary": "..."\n'
        '  },\n'
        '  "strengths_weaknesses_matrix": [\n'
        + ",\n".join(
            '    {"work_id": "' + wid + '", "title": "' + titles_map[wid][:50] + '", '
            '"handles_well": [{"capability": "...", "mechanism": "...", "evidence": "..."}], '
            '"struggles_with": [{"limitation": "...", "cause": "...", "consequence": "..."}], '
            '"assumptions": [{"assumption": "...", "if_violated": "..."}], '
            '"complemented_by": [{"other_work_id": "' + (work_id_list[0] if wid != work_id_list[0] else work_id_list[1] if len(work_id_list) > 1 else wid) + '", "coverage": "..."}]}'
            for wid in work_id_list
        )
        + "\n  ],\n"
        '  "recommendation": {\n'
        '    "summary": "Your quick take on which paper to use and when",\n'
        '    "decision_matrix": [\n'
        '      {"scenario": "...", "use": "' + work_id_list[0] + '", "why": "..."},\n'
        '      {"scenario": "...", "use": "' + (work_id_list[1] if len(work_id_list) > 1 else work_id_list[0]) + '", "why": "..."},\n'
        '      {"scenario": "...", "use": "...", "why": "..."}\n'
        '    ],\n'
        '    "can_combine": true/false,\n'
        '    "combination_notes": "How to combine these methods, or null if not applicable"\n'
        '  }\n'
        "}"
    )


def _synthesize(
    papers_meta: list[Dict[str, Any]],
    fingerprints: Dict[str, Dict[str, Any]],
    contents: list[PaperContent],
    lineage: CitationLineage,
) -> Dict[str, Any]:
    """Run the synthesis LLM call."""
    prompt = _build_synthesis_prompt(papers_meta, fingerprints, contents, lineage)
    work_ids = [p["work_id"] for p in papers_meta]
    result = _call_llm(
        SYNTHESIS_SYSTEM, prompt,
        validator=_validate_synthesis,
        validator_args=(work_ids,),
    )

    if result:
        return result

    # Default fallback
    return {
        "convergence_divergence": {
            "common_problem": {
                "domain": "Unknown",
                "challenge": "Unable to extract",
                "why_hard": "Unable to extract",
            },
            "paradigms": [],
            "divergence_summary": "Unable to generate comparison",
        },
        "strengths_weaknesses_matrix": [],
        "recommendation": {
            "summary": "Unable to generate recommendation",
            "decision_matrix": [],
            "can_combine": False,
            "combination_notes": None,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_comparison_pipeline(
    conn: Connection,
    tenant_id: UUID,
    work_ids: list[str],
    papers_meta_map: Dict[str, Dict[str, Any]],
    map_id: Optional[str] = None,
) -> MethodologyComparisonResponse:
    """
    Shared comparison pipeline (post-validation).

    Steps:
    1. Check comparison cache
    2. Stage 1A + 1B: Fetch content + build lineage (parallel)
    3. Stage 2: Extract fingerprints (LLM call 1)
    4. Stage 3: Synthesize comparison (LLM call 2)
    5. Cache and return
    """
    papers_meta = [papers_meta_map[wid] for wid in work_ids]

    # 1. Check comparison cache
    cached = _get_cached_comparison(conn, work_ids)
    if cached:
        logger.info(f"Comparison cache hit for {work_ids}")
        return MethodologyComparisonResponse(**cached)

    # 2. Stage 1A + 1B (parallel)
    contents: list[PaperContent] = []
    lineage: Optional[CitationLineage] = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_content = pool.submit(fetch_papers_content, conn, work_ids)
        future_lineage = pool.submit(
            build_citation_lineage, conn, work_ids, map_id
        )
        contents = future_content.result()
        lineage = future_lineage.result()

    # 3. Stage 2: Extract fingerprints
    fingerprints = _extract_fingerprints(conn, papers_meta, contents)

    # 4. Stage 3: Synthesize comparison
    synthesis = _synthesize(papers_meta, fingerprints, contents, lineage)

    # 5. Compute confidence
    content_map = {c.work_id: c for c in contents}
    full_text_count = sum(
        1 for c in contents if c.source_quality == "full_text"
    )
    if full_text_count == len(work_ids):
        confidence = "high"
    elif full_text_count > 0:
        confidence = "medium"
    else:
        confidence = "low"

    # 6. Assemble response
    paper_profiles = []
    for wid in work_ids:
        meta = papers_meta_map[wid]
        fp_data = fingerprints.get(wid, {})
        sq = content_map[wid].source_quality if wid in content_map else "abstract_only"
        paper_profiles.append(PaperMethodProfile(
            work_id=wid,
            title=meta.get("title", "Unknown"),
            year=meta.get("year"),
            source_quality=sq,
            methodology_fingerprint=MethodologyFingerprint(**fp_data),
        ))

    # Build convergence_divergence from synthesis
    conv_div_data = synthesis.get("convergence_divergence", {})
    common_prob_data = conv_div_data.get("common_problem", {})
    convergence_divergence = ConvergenceDivergence(
        common_problem=CommonProblem(
            domain=common_prob_data.get("domain", "Unknown"),
            challenge=common_prob_data.get("challenge", "Unknown"),
            why_hard=common_prob_data.get("why_hard", "Unknown"),
        ),
        paradigms=[
            Paradigm(
                name=p.get("name", "Unknown"),
                papers=p.get("papers", []),
                mechanism=p.get("mechanism", "Unknown"),
                philosophy=p.get("philosophy", "Unknown"),
            )
            for p in conv_div_data.get("paradigms", [])
            if isinstance(p, dict)
        ],
        divergence_summary=conv_div_data.get("divergence_summary", ""),
    )

    # Build strengths_weaknesses_matrix from synthesis
    sw_matrix_data = synthesis.get("strengths_weaknesses_matrix", [])
    strengths_weaknesses_matrix = []
    for sw in sw_matrix_data:
        if not isinstance(sw, dict):
            continue
        strengths_weaknesses_matrix.append(PaperStrengthsWeaknesses(
            work_id=sw.get("work_id", ""),
            title=sw.get("title", ""),
            handles_well=[
                Capability(
                    capability=h.get("capability", ""),
                    mechanism=h.get("mechanism", ""),
                    evidence=h.get("evidence", ""),
                )
                for h in sw.get("handles_well", [])
                if isinstance(h, dict)
            ],
            struggles_with=[
                Limitation(
                    limitation=s.get("limitation", ""),
                    cause=s.get("cause", ""),
                    consequence=s.get("consequence", ""),
                )
                for s in sw.get("struggles_with", [])
                if isinstance(s, dict)
            ],
            assumptions=[
                Assumption(
                    assumption=a.get("assumption", ""),
                    if_violated=a.get("if_violated", ""),
                )
                for a in sw.get("assumptions", [])
                if isinstance(a, dict)
            ],
            complemented_by=[
                Complement(
                    other_work_id=c.get("other_work_id", ""),
                    coverage=c.get("coverage", ""),
                )
                for c in sw.get("complemented_by", [])
                if isinstance(c, dict)
            ],
        ))

    # Build recommendation from synthesis
    rec_data = synthesis.get("recommendation", {})
    if isinstance(rec_data, str):
        # Fallback if recommendation is a string (old format)
        recommendation = Recommendation(
            summary=rec_data,
            decision_matrix=[],
            can_combine=False,
            combination_notes=None,
        )
    else:
        recommendation = Recommendation(
            summary=rec_data.get("summary", ""),
            decision_matrix=[
                DecisionScenario(
                    scenario=d.get("scenario", ""),
                    use=d.get("use", ""),
                    why=d.get("why", ""),
                )
                for d in rec_data.get("decision_matrix", [])
                if isinstance(d, dict)
            ],
            can_combine=rec_data.get("can_combine", False),
            combination_notes=rec_data.get("combination_notes"),
        )

    response = MethodologyComparisonResponse(
        work_ids=work_ids,
        papers=paper_profiles,
        lineage=lineage,
        convergence_divergence=convergence_divergence,
        strengths_weaknesses_matrix=strengths_weaknesses_matrix,
        recommendation=recommendation,
        confidence=confidence,
    )

    # 7. Cache
    _cache_comparison(conn, tenant_id, work_ids, response.model_dump())

    return response


def compare_methodologies(
    engine: Engine,
    tenant_id: UUID,
    map_id: str,
    work_ids: list[str],
) -> MethodologyComparisonResponse:
    """Compare methodologies of 2-4 papers from a citation map."""
    with engine.connect() as conn:
        papers_meta_map = _validate_work_ids_in_map(conn, map_id, work_ids)
        return _run_comparison_pipeline(
            conn, tenant_id, work_ids, papers_meta_map, map_id=map_id
        )


def compare_methodologies_for_rank_job(
    engine: Engine,
    tenant_id: UUID,
    rank_job_id: str,
    work_ids: list[str],
) -> MethodologyComparisonResponse:
    """Compare methodologies of 2-4 papers from a rank job's results."""
    with engine.connect() as conn:
        papers_meta_map = _validate_work_ids_in_rank_results(
            conn, rank_job_id, work_ids
        )
        return _run_comparison_pipeline(
            conn, tenant_id, work_ids, papers_meta_map, map_id=None
        )
