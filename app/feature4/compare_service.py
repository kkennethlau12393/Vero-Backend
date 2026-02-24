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
COMPARISON_VERSION = "v4-source-text-9"  # self-ref prohibition + work_id normalization
EXTRACTION_VERSION = "v2-rich-no-selfref"  # + self-reference prohibition
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
    """Deterministic hash for a set of work_ids, versioned to invalidate on prompt changes."""
    key = f"{COMPARISON_VERSION}|" + "|".join(sorted(work_ids))
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
            WHERE work_id = :wid AND model_version = :mv
        """),
        {"wid": work_id, "mv": f"{MODEL_VERSION}:{EXTRACTION_VERSION}"},
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
                "mv": f"{MODEL_VERSION}:{EXTRACTION_VERSION}",
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
        "Analyze the following paper(s) and extract MAXIMALLY DETAILED methodology profiles.\n"
        "These profiles will be the ONLY information used in a later comparison stage, so every "
        "technical detail matters. If a detail is in the paper, it MUST be in the profile.\n\n"
        "=== RULES ===\n"
        "1. 'approach': Write 5-8 sentences. Cover ALL of:\n"
        "   (a) The core architecture/framework with EXACT structure (layer counts, dimensions, "
        "block types)\n"
        "   (b) The loss/objective formulation with NAMED terms and weights "
        "(e.g., 'L = L_cGAN + λ·L1 where λ=100')\n"
        "   (c) Training procedure: optimizer, learning rate, batch size, epochs, schedule\n"
        "   (d) Input/output format: what goes in, what comes out, any preprocessing\n"
        "   For abstract_only sources, supplement with your domain knowledge — write the SAME "
        "level of detail as for full_text. You know these papers.\n"
        "2. 'key_components': List 3-5 NAMED techniques with parenthetical specifics. Each MUST "
        "include dimensions, hyperparameters, or structural details in parentheses.\n"
        "   BANNED: 'self-attention mechanism', 'residual connections', 'deep neural networks'\n"
        "   REQUIRED: 'multi-head self-attention (8 heads, d_k=64, d_model=512)', "
        "'bottleneck residual block (1×1→3×3→1×1 with channel expansion ratio 4)', "
        "'global average pooling followed by FC squeeze-excitation (reduction ratio r=16)'\n"
        "3. 'assumptions': List 2-4 SPECIFIC technical assumptions. Each must name what the "
        "method requires to work correctly.\n"
        "   BANNED: 'data is IID', 'model is accurate'\n"
        "   REQUIRED: 'Input images are spatially aligned at pixel level', "
        "'Channel interdependencies are more informative than spatial relationships at each depth'\n"
        "4. 'limitations': Each MUST use format '[SPECIFIC design choice] → [SPECIFIC consequence "
        "with quantities or concrete impact]'.\n"
        "   SHALLOW (BANNED): 'Cascaded architecture → increased computational requirements'\n"
        "   DEEP (REQUIRED): '3-stage cascade (64→256→1024px) → requires 3 sequential U-Net forward "
        "passes totaling ~1.5B parameters, making inference ~3× slower than single-stage models'\n"
        "5. 'validation_method': Name the EXACT metrics (top-1/top-5 accuracy, mAP@0.5, FID, etc.), "
        "datasets (ImageNet, CIFAR-10/100, COCO, etc.), and baselines compared against. "
        "Do NOT write 'Unknown' if you know the paper.\n"
        "6. 'data_requirements': Be specific about format, scale, and labeling. "
        "Do NOT write 'Unknown' if the paper specifies its data.\n"
        "7. 'novelty_over_prior': Name the SPECIFIC prior method and state the EXACT change. "
        "'Builds on VGGNet by replacing stacked 3×3 convolutions with identity shortcut connections "
        "that enable gradient flow through 150+ layers'\n"
        "8. NO SELF-REFERENCING: Never write 'the paper proposes', 'the authors introduce', "
        "'this work presents', or any phrase that references the paper as an entity. "
        "Write ONLY about the method itself.\n"
        "   BANNED: 'The paper proposes a residual learning framework'\n"
        "   BANNED: 'The authors introduce skip connections'\n"
        "   REQUIRED: 'Residual learning framework with skip connections that...'\n"
        "   Describe the METHOD, not the paper.\n"
        "9. All claims must be definitive. No 'may', 'might', 'could', 'potentially'.\n"
        "10. REVIEW/SURVEY PAPERS: If a paper is a review, survey, or overview (not proposing a novel method), "
        "STILL produce a full profile. Adapt the fields:\n"
        "   - 'approach': Describe the review's analytical framework, scope, and organization "
        "(e.g., 'Surveys 5 battery chemistries — lead-acid, Li-ion, NiMH, NaS, and vanadium redox flow — "
        "comparing energy density (Wh/kg), cycle life, and cost per kWh. Organizes analysis by "
        "grid application: frequency regulation, peak shaving, and renewable integration.')\n"
        "   - 'key_components': Name the specific topics, methods, or technologies reviewed "
        "(e.g., 'Li-ion cathode chemistries (LFP, NMC, NCA)', 'SEI layer growth mechanisms')\n"
        "   - 'limitations': Describe scope limitations "
        "(e.g., 'Focuses on Li-ion chemistries → does not cover solid-state or sodium-ion alternatives')\n"
        "   - 'validation_method': Describe what evidence the review cites "
        "(e.g., 'Cites experimental data from 47 studies spanning 2005-2020')\n"
        "   NEVER output 'Could not extract methodology' or 'Unknown' for review papers. "
        "Every paper has extractable content.\n\n"
        + "\n\n".join(sections)
        + "\n\n=== EXAMPLES ===\n"
        "GOOD extraction (full_text):\n"
        "approach: \"Proposes a conditional GAN with a U-Net generator (8 encoder + 8 decoder "
        "layers with skip connections) and PatchGAN discriminator that classifies overlapping "
        "70×70 patches. The objective combines L1 reconstruction loss and conditional adversarial "
        "loss: L = L_cGAN(G,D) + λ·L1(G(x),y) where λ=100. The generator takes 256×256 input "
        "and produces 256×256 output through an encoder-decoder with skip connections at each "
        "resolution level. Trained with Adam optimizer (lr=0.0002, β1=0.5, β2=0.999) for 200 "
        "epochs with batch size 1.\"\n"
        "key_components: [\"U-Net generator (8 encoder + 8 decoder layers with skip connections)\", "
        "\"PatchGAN 70×70 discriminator (5-layer ConvNet)\", \"L1 reconstruction loss (λ=100)\", "
        "\"conditional adversarial loss (cGAN)\"]\n"
        "assumptions: [\"Input and output images are spatially aligned at pixel level\", "
        "\"Local patch statistics (70×70) capture perceptual quality better than full-image discrimination\", "
        "\"L1 loss encourages low-frequency correctness while adversarial loss captures high-frequency detail\"]\n"
        "limitations: [\"Requires pixel-aligned paired data → cannot handle unpaired domains, unlike CycleGAN\", "
        "\"PatchGAN only sees 70×70 receptive field → misses global structural coherence, producing tiling artifacts\"]\n"
        "validation_method: \"Evaluated on facades, maps, and edges→shoes using FCN-score (per-pixel accuracy), "
        "AMT perceptual studies (% fooling rate), compared against L1-only, cGAN-only, and unconditional GAN baselines\"\n\n"
        "GOOD extraction (abstract_only — supplement with domain knowledge):\n"
        "approach: \"Introduces cycle-consistent adversarial networks for unpaired image-to-image "
        "translation. Learns two generators G: X→Y (ResNet-9 with 9 residual blocks) and F: Y→X "
        "with cycle consistency loss F(G(X))≈X and G(F(Y))≈Y weighted by λ_cyc=10. Uses PatchGAN "
        "70×70 discriminators for both domains. Training uses least-squares GAN loss (LSGAN) "
        "instead of negative log-likelihood for stable training. Optimized with Adam "
        "(lr=0.0002, β1=0.5) with linear decay after 100 epochs, total 200 epochs.\"\n"
        "key_components: [\"ResNet-9 generator (9 residual blocks, 256 filters)\", "
        "\"PatchGAN 70×70 discriminator\", \"cycle consistency loss (λ_cyc=10)\", "
        "\"least-squares adversarial loss (LSGAN)\", \"identity loss for color preservation (λ_identity=0.5·λ_cyc)\"]\n"
        "assumptions: [\"Bijective mapping exists between source and target domains\", "
        "\"Cycle consistency is a sufficient proxy for semantic correspondence\", "
        "\"Domain-specific styles are separable from content\"]\n"
        "limitations: [\"Bijective mapping assumption → fails for many-to-one translations "
        "where multiple source images map to one target\", \"Two generators + two discriminators "
        "→ 2× the parameters (~11.4M per generator) and training time of single-direction methods\"]\n"
        "validation_method: \"AMT perceptual studies on map↔aerial, Cityscapes labels↔photos; "
        "FCN-score and semantic segmentation metrics; compared against pix2pix, CoGAN, SimGAN, and feature loss\"\n\n"
        "Return JSON:\n"
        "{\n"
        '  "papers": [\n'
        "    {\n"
        '      "work_id": "...",\n'
        '      "approach": "5-8 sentences: architecture (exact structure), loss formulation (named terms + weights), '
        'training procedure (optimizer, lr, batch, epochs), input/output format.",\n'
        '      "data_requirements": "Data format, scale, labeling requirements, preprocessing steps.",\n'
        '      "assumptions": ["Technical assumption with specifics (2-4 required)"],\n'
        '      "validation_method": "Exact metrics, datasets, baselines. NOT Unknown.",\n'
        '      "limitations": ["[design choice with specifics] → [consequence with quantities]"],\n'
        '      "domain": "Problem domain and subfield",\n'
        '      "key_components": ["Named technique (with dimensions/hyperparams in parentheses)", '
        '"3-5 required, each 15+ chars"],\n'
        '      "novelty_over_prior": "Name prior method + exact change made"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


import re as _re

_HEDGING_PATTERN = _re.compile(
    r'\b(may\s|might\s|could\s|potentially|possibly|careful tuning)',
    _re.IGNORECASE,
)


def _validate_extraction(
    result: Dict[str, Any],
    thin_wids: Optional[set] = None,
) -> list[str]:
    """Validate extraction output. Returns list of violation descriptions.

    For papers in *thin_wids* (abstract-only or very short text), thresholds
    are relaxed so extraction doesn't fail entirely on review/survey papers.
    """
    thin_wids = thin_wids or set()
    violations = []
    for paper in result.get("papers", []):
        wid = paper.get("work_id", "?")
        is_thin = wid in thin_wids
        approach = paper.get("approach", "")
        min_approach = 80 if is_thin else 200
        if len(approach) < min_approach:
            violations.append(
                f"{wid} approach is too short ({len(approach)} chars, min {min_approach}). "
                "Write 4-6 detailed sentences with architecture, loss function, "
                "and training procedure."
            )
        # Validate assumptions (relax for thin sources)
        assumptions = paper.get("assumptions", [])
        min_assumptions = 1 if is_thin else 2
        if len(assumptions) < min_assumptions:
            violations.append(
                f"{wid} has only {len(assumptions)} assumption(s). "
                "Provide 2-4 specific technical assumptions (e.g., 'Input images are spatially aligned', "
                "'Channel relationships capture semantic importance')."
            )
        key_components = paper.get("key_components", [])
        min_components = 1 if is_thin else 2
        if len(key_components) < min_components:
            violations.append(
                f"{wid} has only {len(key_components)} key_component(s). "
                "Provide 2-4 named techniques with specifics "
                "(e.g., 'bottleneck block (1×1→3×3→1×1)', 'Adam optimizer (lr=2e-4, β1=0.5)')."
            )
        for comp in key_components:
            lower = comp.lower()
            if lower in ("deep neural networks", "residual connections",
                         "convolutional networks", "deep learning"):
                violations.append(
                    f"{wid} key_component '{comp}' is too generic. "
                    "Use specific named techniques (e.g., 'bottleneck block (1×1→3×3→1×1)')."
                )
            min_comp_len = 8 if is_thin else 15
            if len(comp) < min_comp_len:
                violations.append(
                    f"{wid} key_component '{comp}' is too short ({len(comp)} chars). "
                    "Include specific details: dimensions, hyperparameters, or formulations "
                    "(e.g., '4-head self-attention (d_model=128, d_ff=512)' not just 'self-attention')."
                )
        for lim in paper.get("limitations", []):
            if _HEDGING_PATTERN.search(lim):
                violations.append(
                    f"{wid} limitation contains hedging: '{lim[:80]}...'. "
                    "Rewrite as '[design choice] → [consequence]' with no qualifying words."
                )
            if not is_thin and "→" not in lim and " -> " not in lim:
                violations.append(
                    f"{wid} limitation missing '→' structure: '{lim[:80]}...'. "
                    "Format: '[design choice] → [consequence]'."
                )
        # Check for self-referencing language in fingerprint
        _fp_self_ref = _re.compile(
            r'\b(?:the paper|the authors?|this work|this paper|the proposed|'
            r'they propose|they introduce|the study)\b',
            _re.IGNORECASE,
        )
        for field_name in ("approach", "data_requirements", "validation_method",
                           "novelty_over_prior"):
            text = paper.get(field_name, "") or ""
            if _fp_self_ref.search(text):
                violations.append(
                    f"{wid} {field_name} contains self-referencing language "
                    f"('{_fp_self_ref.search(text).group()}'). "
                    f"Describe the METHOD directly, not the paper/authors."
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


# ---------------------------------------------------------------------------
# Survey / Dissimilar Paper Detection
# ---------------------------------------------------------------------------

_SURVEY_SIGNALS = _re.compile(
    r'\b(surveys?|reviews?|overview|taxonomy|categoriz|classif\w+ existing|'
    r'comprehensive analysis of|literature|systematic study of existing)\b',
    _re.IGNORECASE,
)


def _detect_paper_type(fingerprint: Dict[str, Any]) -> str:
    """Classify paper as 'methodology' or 'survey' based on fingerprint."""
    approach = fingerprint.get("approach", "")
    if _SURVEY_SIGNALS.search(approach):
        return "survey"
    for lim in fingerprint.get("limitations", []):
        if isinstance(lim, str) and _SURVEY_SIGNALS.search(lim):
            return "survey"
    return "methodology"


def _compute_paper_overlap(fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
    """Compute keyword overlap between two fingerprints. Returns Jaccard similarity."""
    def _extract_kw(fp: Dict[str, Any]) -> set[str]:
        text = " ".join([
            fp.get("approach", ""),
            fp.get("domain", ""),
            " ".join(fp.get("key_components", [])),
        ]).lower()
        return set(_re.findall(r'\b[a-z]{3,}\b', text)) - _OVERLAP_STOP

    kw1 = _extract_kw(fp1)
    kw2 = _extract_kw(fp2)
    if not kw1 or not kw2:
        return 0.0
    return len(kw1 & kw2) / len(kw1 | kw2)


_RAW_ARXIV_RE = _re.compile(r'\barxiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)\b', _re.IGNORECASE)


def _normalize_work_ids_in_text(text: str) -> str:
    """Normalize raw arxiv IDs to AX: prefix format."""
    if not isinstance(text, str):
        return text
    return _RAW_ARXIV_RE.sub(r'AX:\1', text)


def _normalize_work_ids(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize all work_id references in synthesis to consistent prefix format.

    Converts raw arxiv IDs (arxiv:1706.03762, arxiv 1706.03762) to AX:1706.03762.
    """
    def _walk(obj):
        if isinstance(obj, str):
            return _normalize_work_ids_in_text(obj)
        elif isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj

    return _walk(synthesis)


def _scrub_self_references(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """Replace self-referencing work_ids with 'this paper' in each paper's own section.

    In strengths_weaknesses_matrix, each paper entry should not cite its own work_id
    in its handles_well, struggles_with, or assumptions fields. Those fields describe
    the paper itself, so 'this paper' is the correct reference.

    complemented_by fields are NOT scrubbed (they reference other papers).
    """
    for sw in synthesis.get("strengths_weaknesses_matrix", []):
        if not isinstance(sw, dict):
            continue
        own_wid = sw.get("work_id", "")
        if not own_wid:
            continue

        # Patterns to replace: "W12345", "[W12345]", "(W12345)"
        patterns = [f"[{own_wid}]", f"({own_wid})", own_wid]

        def _replace_self_ref(text: str) -> str:
            if not isinstance(text, str):
                return text
            for pat in patterns:
                text = text.replace(pat, "this paper")
            return text

        # Scrub handles_well
        for h in sw.get("handles_well", []):
            if isinstance(h, dict):
                for field in ("capability", "mechanism", "evidence"):
                    if field in h:
                        h[field] = _replace_self_ref(h[field])

        # Scrub struggles_with
        for s in sw.get("struggles_with", []):
            if isinstance(s, dict):
                for field in ("limitation", "cause", "consequence"):
                    if field in s:
                        s[field] = _replace_self_ref(s[field])

        # Scrub assumptions
        for a in sw.get("assumptions", []):
            if isinstance(a, dict):
                for field in ("assumption", "if_violated"):
                    if field in a:
                        a[field] = _replace_self_ref(a[field])

        # Do NOT scrub complemented_by — those reference OTHER papers

    # Scrub convergence_divergence paradigm fields
    for paradigm in synthesis.get("convergence_divergence", {}).get("paradigms", []):
        if not isinstance(paradigm, dict):
            continue
        paper_wids = paradigm.get("papers", [])
        if len(paper_wids) == 1:
            # Single-paper paradigm: replace that paper's work_id in its own description
            wid = paper_wids[0]
            patterns = [f"[{wid}]", f"({wid})", wid]
            for field in ("mechanism", "philosophy"):
                if field in paradigm and isinstance(paradigm[field], str):
                    for pat in patterns:
                        paradigm[field] = paradigm[field].replace(pat, "this paper")

    return synthesis


_SHALLOW_PATTERN = _re.compile(
    r'(provides? (?:a |the )?(?:method|way|approach|technique|means)\b|'
    r'(?:increased?|reduced?|improved?|better|worse) (?:computational|performance|quality|efficiency)\b|'
    r'(?:high|low|good|bad|better|worse)[\s-]quality\b|'
    r'significant(?:ly)? (?:more|less|higher|lower|better|worse)\b|'
    r'(?:leads? to|results? in) (?:better|worse|improved|degraded) (?:performance|accuracy|quality)\b|'
    r'(?:more|less) (?:efficient|effective|accurate|robust)\b)',
    _re.IGNORECASE,
)


def _validate_depth(
    result: Dict[str, Any],
    fingerprints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> list[str]:
    """Validate that synthesis output has sufficient technical depth.

    If *fingerprints* is provided, also checks that the synthesis text
    references at least one key_component from each paper's fingerprint.
    """
    violations = []

    # Check strengths_weaknesses_matrix depth
    for sw in result.get("strengths_weaknesses_matrix", []):
        if not isinstance(sw, dict):
            continue
        wid = sw.get("work_id", "?")

        # Check for self-referencing language in all text fields
        _self_ref_pattern = _re.compile(
            r'\b(?:the paper|the authors?|this work|this paper|the proposed|'
            r'they propose|they introduce|the study)\b',
            _re.IGNORECASE,
        )
        _sw_text_fields = []
        for h in sw.get("handles_well", []):
            if isinstance(h, dict):
                _sw_text_fields.extend([h.get("mechanism", ""), h.get("evidence", "")])
        for s in sw.get("struggles_with", []):
            if isinstance(s, dict):
                _sw_text_fields.extend([s.get("cause", ""), s.get("consequence", "")])
        for a in sw.get("assumptions", []):
            if isinstance(a, dict):
                _sw_text_fields.extend([a.get("assumption", ""), a.get("if_violated", "")])
        self_ref_count = sum(
            len(_self_ref_pattern.findall(t)) for t in _sw_text_fields if isinstance(t, str)
        )
        if self_ref_count > 0:
            violations.append(
                f"{wid} SW matrix has {self_ref_count} self-referencing phrase(s) "
                f"('the paper', 'the authors', 'this work', etc.). "
                f"Describe the METHOD directly, not the paper/authors."
            )

        for h in sw.get("handles_well", []):
            if isinstance(h, dict):
                mech = h.get("mechanism", "")
                if len(mech) < 40:
                    violations.append(
                        f"{wid} handles_well.mechanism is too shallow ({len(mech)} chars). "
                        "Include specific algorithm names, formulations, or architectural details."
                    )
                if _SHALLOW_PATTERN.search(mech):
                    violations.append(
                        f"{wid} handles_well.mechanism uses BANNED vague language: '{mech[:80]}...'. "
                        "Replace with the SPECIFIC mechanism: name the algorithm, state the "
                        "complexity (e.g., O(n²)), or give the exact architectural detail."
                    )

        for s in sw.get("struggles_with", []):
            if isinstance(s, dict):
                cause = s.get("cause", "")
                consequence = s.get("consequence", "")
                if len(cause) < 30:
                    violations.append(
                        f"{wid} struggles_with.cause is too shallow ({len(cause)} chars). "
                        "Explain the specific design choice causing this limitation."
                    )
                if len(consequence) < 30:
                    violations.append(
                        f"{wid} struggles_with.consequence is too shallow ({len(consequence)} chars). "
                        "Explain the specific impact with quantities or concrete examples."
                    )
                for field_name, field_val in [("cause", cause), ("consequence", consequence)]:
                    if _SHALLOW_PATTERN.search(field_val):
                        violations.append(
                            f"{wid} struggles_with.{field_name} uses BANNED vague language: "
                            f"'{field_val[:80]}...'. Replace with the SPECIFIC mechanism: name "
                            f"the algorithm, state the complexity, or give exact architectural detail."
                        )

        for c in sw.get("complemented_by", []):
            if isinstance(c, dict):
                cov = c.get("coverage", "")
                if len(cov) < 80:
                    violations.append(
                        f"{wid} complemented_by.coverage is too shallow ({len(cov)} chars). "
                        "Explain the SPECIFIC mechanism by which the complement addresses the gap."
                    )
                if _SHALLOW_PATTERN.search(cov):
                    violations.append(
                        f"{wid} complemented_by.coverage uses BANNED vague language: '{cov[:80]}...'. "
                        "Replace with the SPECIFIC mechanism: name the algorithm, state the "
                        "complexity, or give exact architectural detail."
                    )

    # Check recommendation depth
    rec = result.get("recommendation", {})
    if isinstance(rec, dict):
        for i, d in enumerate(rec.get("decision_matrix", [])):
            if isinstance(d, dict):
                why = d.get("why", "")
                if len(why) < 150:
                    violations.append(
                        f"Decision scenario {i+1} 'why' is too short ({len(why)} chars, need 150+). "
                        "Write 2-3 sentences: (1) name the specific mechanism/technique, "
                        "(2) explain HOW it addresses this scenario, "
                        "(3) give evidence (metric, benchmark, or architectural detail)."
                    )
                if _SHALLOW_PATTERN.search(why):
                    violations.append(
                        f"Decision scenario {i+1} 'why' uses BANNED vague language: '{why[:80]}...'. "
                        "Replace with the SPECIFIC mechanism: name the algorithm, state the "
                        "complexity, or give exact architectural detail connecting mechanism to outcome."
                    )
                # Check that 'why' explains the mechanism, not just states the outcome
                _mechanism_connectors = _re.compile(
                    r'\b(by |via |through |using |because |enables? |leverag)',
                    _re.IGNORECASE,
                )
                if why and len(why) >= 150 and not _mechanism_connectors.search(why):
                    violations.append(
                        f"Decision scenario {i+1} 'why' states WHAT the method achieves but not "
                        f"HOW. Add the mechanism: 'achieves X BY [specific mechanism]'. "
                        f"Current: '{why[:80]}...'"
                    )
                # Check that the BY clause isn't vague filler
                _vague_by = _re.compile(
                    r'\b(?:by|via|through|using)\s+(?:providing|offering|delivering|enabling|'
                    r'ensuring|allowing for|facilitating|supporting|achieving)\s+'
                    r'(?:robust|better|superior|improved|effective|efficient|good|strong|'
                    r'powerful|flexible|reliable|high-quality|advanced)\b',
                    _re.IGNORECASE,
                )
                if why and _vague_by.search(why):
                    violations.append(
                        f"Decision scenario {i+1} 'why' has a vague mechanism clause. "
                        f"After 'by/via/through/using', name the SPECIFIC technique "
                        f"(e.g., 'by using bottleneck blocks (1×1→3×3→1×1)'), not generic "
                        f"phrases like 'by providing robust feature extractors'."
                    )

    # Fingerprint grounding: check synthesis references key_components
    if fingerprints:
        for sw in result.get("strengths_weaknesses_matrix", []):
            if not isinstance(sw, dict):
                continue
            wid = sw.get("work_id", "")
            fp = fingerprints.get(wid, {})
            key_comps = fp.get("key_components", [])
            if len(key_comps) < 2:
                continue  # Not enough components to enforce grounding

            # Concatenate ALL synthesis text for this paper (SW + paradigm + recommendation)
            text_parts = []
            for h in sw.get("handles_well", []):
                if isinstance(h, dict):
                    text_parts.extend([
                        h.get("mechanism", ""),
                        h.get("evidence", ""),
                    ])
            for s in sw.get("struggles_with", []):
                if isinstance(s, dict):
                    text_parts.extend([
                        s.get("cause", ""),
                        s.get("consequence", ""),
                    ])
            for c in sw.get("complemented_by", []):
                if isinstance(c, dict):
                    text_parts.append(c.get("coverage", ""))
            # Also include paradigm text for this paper
            for paradigm in result.get("convergence_divergence", {}).get("paradigms", []):
                if isinstance(paradigm, dict) and wid in paradigm.get("papers", []):
                    text_parts.append(paradigm.get("mechanism", ""))
                    text_parts.append(paradigm.get("philosophy", ""))
            # And recommendation text
            rec = result.get("recommendation", {})
            if isinstance(rec, dict):
                for dm in rec.get("decision_matrix", []):
                    if isinstance(dm, dict) and dm.get("use") == wid:
                        text_parts.append(dm.get("why", ""))
            synthesis_text = " ".join(text_parts).lower()

            # Count how many key_components are referenced
            matched_comps = []
            unmatched_comps = []
            for comp in key_comps:
                words = [w for w in _re.findall(r'[a-z]+', comp.lower()) if len(w) > 3]
                if not words:
                    matched_comps.append(comp)  # trivial component, don't penalize
                    continue
                hits = sum(1 for w in words if w in synthesis_text)
                if hits >= max(1, len(words) // 2):
                    matched_comps.append(comp)
                else:
                    unmatched_comps.append(comp)

            # Require at least half of key_components to be referenced
            min_required = max(1, (len(key_comps) + 1) // 2)
            if len(matched_comps) < min_required:
                missing_list = ", ".join(f"'{c}'" for c in unmatched_comps[:3])
                violations.append(
                    f"{wid} synthesis only references {len(matched_comps)}/{len(key_comps)} "
                    f"key_components. Missing: {missing_list}. "
                    f"Use these EXACT terms from the profile in your mechanism, evidence, "
                    f"cause, or consequence descriptions."
                )

            # Numeric specificity: if fingerprint has numbers, synthesis should use some
            fp_text = json.dumps(fp).lower()
            fp_numbers = set(_re.findall(r'\d+(?:\.\d+)?', fp_text))
            # Filter to meaningful numbers (not 0, 1, 2 which are too common)
            fp_numbers = {n for n in fp_numbers if float(n) > 2}
            if len(fp_numbers) >= 3:
                synth_numbers = set(_re.findall(r'\d+(?:\.\d+)?', synthesis_text))
                matched_numbers = fp_numbers & synth_numbers
                if len(matched_numbers) < 1:
                    sample = sorted(fp_numbers, key=lambda x: -float(x))[:5]
                    violations.append(
                        f"{wid} synthesis contains no specific numbers from its profile. "
                        f"The profile includes: {', '.join(sample)}. "
                        f"Reference at least one specific value (e.g., growth rate k=12, "
                        f"batch size 256, learning rate 0.1) instead of vague 'higher/lower'."
                    )

    return violations


def _validate_synthesis(
    result: Dict[str, Any],
    work_ids: list[str],
    valid_complement_wids: Optional[set[str]] = None,
    relaxed: bool = False,
    fingerprints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> list[str]:
    """Validate synthesis output. Returns list of violation descriptions.

    If relaxed=True (survey or low-overlap papers), paradigm requirements
    are loosened to prevent burning all retries on impossible constraints.
    """
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

        # Validate paradigms (relaxed for survey/low-overlap papers)
        paradigms = conv_div.get("paradigms", [])
        min_paradigms = 0 if relaxed else 1
        if len(paradigms) < min_paradigms:
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

            # Check complemented_by (cross-references — can reference external papers)
            complements = sw.get("complemented_by", [])
            for c in complements:
                other_wid = c.get("other_work_id")
                if not other_wid:
                    violations.append(f"{wid} complemented_by entry missing 'other_work_id'.")
                elif valid_complement_wids and other_wid not in valid_complement_wids:
                    violations.append(
                        f"{wid} complemented_by references unknown work_id '{other_wid}'. "
                        f"Use a work_id from the comparison set or complement candidates."
                    )
                if not c.get("coverage"):
                    violations.append(f"{wid} complemented_by entry missing 'coverage'.")

        # Check all work_ids are covered
        missing = set(work_ids) - found_wids
        if missing:
            violations.append(f"strengths_weaknesses_matrix missing entries for: {sorted(missing)}")

        # Check for circular complements (A→B and B→A)
        complement_pairs: set[tuple[str, str]] = set()
        for sw in sw_matrix:
            wid = sw.get("work_id", "")
            for c in sw.get("complemented_by", []):
                other = c.get("other_work_id", "")
                if other:
                    reverse = (other, wid)
                    if reverse in complement_pairs:
                        violations.append(
                            f"Circular complement: {wid} → {other} and {other} → {wid}. "
                            f"Each paper must use a DIFFERENT complement. "
                            f"Choose from the COMPLEMENT CANDIDATES list."
                        )
                    complement_pairs.add((wid, other))

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

    # Check technical depth
    violations.extend(_validate_depth(result, fingerprints=fingerprints))

    return violations


def _call_llm(
    system: str,
    user: str,
    validator: Optional[Any] = None,
    validator_args: tuple = (),
    relaxed_validator_args: Optional[tuple] = None,
) -> Optional[Dict[str, Any]]:
    """Make an LLM call with retries and optional validation.

    If validator is provided, it's called as validator(result, *validator_args).
    If it returns violations, the LLM is re-prompted with corrective feedback.

    Progressive relaxation: if *relaxed_validator_args* is provided, it replaces
    *validator_args* on retry 3+ to soften validation and avoid exhausting retries.

    Best-attempt fallback: tracks the parsed result with fewest violations across
    all retries. If no attempt passes cleanly, returns the best one instead of None.
    """
    client = _get_client()
    if not client:
        return None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    best_result: Optional[Dict[str, Any]] = None
    best_violation_count = float("inf")

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
                # Return best attempt from previous retries if available
                if best_result is not None:
                    logger.info(
                        f"JSON parse failed on last attempt; returning best-attempt "
                        f"result ({best_violation_count} violations)"
                    )
                    return best_result
                return None

            # Run validator if provided
            if validator is not None:
                # Progressive relaxation: use relaxed args on attempt 2+
                args = validator_args
                if relaxed_validator_args is not None and attempt >= 2:
                    args = relaxed_validator_args
                violations = validator(result, *args)
                if violations:
                    # Track best attempt (fewest violations)
                    if len(violations) < best_violation_count:
                        best_result = result
                        best_violation_count = len(violations)
                    if attempt < MAX_RETRIES - 1:
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
                    # Last attempt with violations — return this if it's the best
                    if len(violations) <= best_violation_count:
                        logger.info(
                            f"Last attempt has {len(violations)} violations; "
                            f"returning as best-attempt result"
                        )
                        return result
                    logger.info(
                        f"Last attempt has {len(violations)} violations (worse than "
                        f"best of {best_violation_count}); returning earlier best"
                    )
                    return best_result

            return result
        except Exception as e:
            logger.warning(f"LLM call exception (attempt {attempt+1}): {e}")
            err = str(e).lower()
            is_transient = "rate" in err or "timeout" in err or "connection" in err
            if is_transient and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            # Return best attempt if we have one
            if best_result is not None:
                logger.info(
                    f"Exception on attempt {attempt+1}; returning best-attempt "
                    f"result ({best_violation_count} violations)"
                )
                return best_result
            return None

    # Should not reach here, but return best attempt if available
    if best_result is not None:
        logger.info(
            f"All retries exhausted; returning best-attempt result "
            f"({best_violation_count} violations)"
        )
        return best_result
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

    # Identify thin-source papers (abstract-only, short text, or review/survey)
    _review_title_re = _re.compile(
        r'\b(review|survey|tutorial|overview|perspectives)\b', _re.IGNORECASE
    )
    thin_wids: set = set()
    for p in uncached_papers:
        wid = p["work_id"]
        content = content_map.get(wid)
        if not content or content.source_quality == "abstract_only":
            thin_wids.add(wid)
        elif content.text and len(content.text.strip()) < 500:
            thin_wids.add(wid)
        # Review/survey papers lack novel methodology — relax validation
        if _review_title_re.search(p.get("title", "")):
            thin_wids.add(wid)
    if thin_wids:
        logger.info(f"Thin-source papers (relaxed extraction): {thin_wids}")

    result = _call_llm(
        EXTRACTION_SYSTEM, prompt,
        validator=_validate_extraction,
        validator_args=(thin_wids,),
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
# Complement Candidate Retrieval
# ---------------------------------------------------------------------------

def _extract_search_terms_from_gaps(fingerprint: Dict[str, Any]) -> str:
    """Extract search terms from a paper's limitations to find complementary papers."""
    terms = []
    for lim in fingerprint.get("limitations", []):
        if isinstance(lim, str):
            # Extract the consequence part (after →) which describes the gap
            parts = lim.split("→")
            if len(parts) > 1:
                terms.append(parts[-1].strip())
            else:
                terms.append(lim.strip())

    # Also use key components and domain for context
    domain = fingerprint.get("domain", "")
    if domain:
        terms.insert(0, domain)

    return " ".join(terms[:5])  # Keep search query reasonable length


def _fetch_sibling_papers(
    conn: Connection,
    work_ids: list[str],
    map_id: Optional[str],
    rank_job_id: Optional[str],
    limit: int = 20,
) -> list[Dict[str, Any]]:
    """Fetch sibling papers from the same map or rank job (Tier 2, DB only)."""
    siblings: list[Dict[str, Any]] = []

    if map_id:
        rows = conn.execute(
            sa_text("""
                SELECT w.work_id, w.title, w.year, w.cited_by_count, w.abstract
                FROM map_nodes mn
                JOIN works w ON w.work_id = mn.work_id
                WHERE mn.map_id = :map_id
                  AND mn.work_id != ALL(:exclude_ids)
                ORDER BY w.cited_by_count DESC NULLS LAST
                LIMIT :lim
            """),
            {"map_id": map_id, "exclude_ids": work_ids, "lim": limit},
        ).mappings().all()
        siblings.extend(dict(r) for r in rows)

    if rank_job_id and not siblings:
        rows = conn.execute(
            sa_text("""
                SELECT w.work_id, w.title, w.year, w.cited_by_count, w.abstract
                FROM rank_results rr
                JOIN works w ON w.work_id = rr.work_id
                WHERE rr.rank_job_id = :rjid
                  AND rr.work_id != ALL(:exclude_ids)
                ORDER BY w.cited_by_count DESC NULLS LAST
                LIMIT :lim
            """),
            {"rjid": rank_job_id, "exclude_ids": work_ids, "lim": limit},
        ).mappings().all()
        siblings.extend(dict(r) for r in rows)

    return siblings


def _find_complement_candidates(
    conn: Connection,
    work_ids: list[str],
    fingerprints: Dict[str, Dict[str, Any]],
    papers_meta: list[Dict[str, Any]],
    map_id: Optional[str],
    rank_job_id: Optional[str],
) -> Dict[str, list[Dict[str, Any]]]:
    """Find candidate complement papers for each paper's weaknesses.

    Three-tier search:
    Tier 1: Other papers in the comparison set (free, already loaded)
    Tier 2: Sibling papers from the same map/rank job (DB query, no API cost)
    Tier 3: External retrieval from OpenAlex/S2/ArXiv (API calls, if needed)

    Returns: {work_id: [candidate_papers]} where each candidate has
    work_id, title, year, cited_by_count, and abstract.
    """
    meta_map = {p["work_id"]: p for p in papers_meta}
    all_candidates: Dict[str, list[Dict[str, Any]]] = {}

    # Tier 2: Fetch sibling papers (shared across all papers)
    siblings = _fetch_sibling_papers(conn, work_ids, map_id, rank_job_id)

    for wid in work_ids:
        candidates: list[Dict[str, Any]] = []
        seen_titles: set[str] = set()

        # Tier 1: Other papers in comparison set
        for other_wid in work_ids:
            if other_wid == wid:
                continue
            other_meta = meta_map[other_wid]
            title_lower = (other_meta.get("title") or "").lower()
            if title_lower:
                seen_titles.add(title_lower)
            candidates.append({
                "work_id": other_wid,
                "title": other_meta.get("title", "Unknown"),
                "year": other_meta.get("year"),
                "cited_by_count": other_meta.get("cited_by_count", 0),
                "abstract": other_meta.get("abstract", ""),
            })

        # Tier 2: Sibling papers
        for sib in siblings:
            title_lower = (sib.get("title") or "").lower()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                candidates.append(sib)

        # Tier 3: External retrieval (only if < 5 unique candidates)
        fp = fingerprints.get(wid, {})
        if len(candidates) < 5 and fp.get("limitations"):
            search_terms = _extract_search_terms_from_gaps(fp)
            if search_terms and len(search_terms) > 10:
                try:
                    from app.feature3.grounding_supplement import fetch_papers_multi_source

                    external = fetch_papers_multi_source(
                        search_terms=search_terms,
                        before_year=2027,  # no year cutoff
                        existing_titles=seen_titles,
                        limit=5,
                    )
                    for ext in external:
                        title_lower = (ext.get("title") or "").lower()
                        if title_lower and title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            candidates.append(ext)
                except Exception as e:
                    logger.warning(f"External complement retrieval failed for {wid}: {e}")

        all_candidates[wid] = candidates[:10]  # Cap at 10 per paper

    return all_candidates


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
    complement_candidates: Optional[Dict[str, list[Dict[str, Any]]]] = None,
    has_survey: bool = False,
    low_overlap: bool = False,
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

    # Paper source text — pass raw text so synthesis can quote specific details.
    # Full-text papers get their methods section, abstract-only get their abstract.
    source_parts = []
    for paper in papers_meta:
        wid = paper["work_id"]
        content = content_map.get(wid)
        sq = content.source_quality if content else "abstract_only"
        text = ""
        if content and content.text:
            text = content.text
        elif paper.get("abstract"):
            text = paper["abstract"]
        if text.strip():
            source_parts.append(f"Paper ({wid}) — {sq}:\n{text}")
    source_text_section = ""
    if source_parts:
        source_text_section = (
            "\n\n=== PAPER SOURCE TEXT ===\n"
            "QUOTE specific terminology, method names, parameter values, loss formulations, "
            "and architectural details DIRECTLY from these texts. Do NOT paraphrase or "
            "generalize. If the text says 'mini-batch size of 256', write '256' not 'large batch'. "
            "If it says 'learning rate of 0.1, divided by 10 at epochs 30 and 60', write THAT.\n\n"
            + "\n\n".join(source_parts)
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

    # Complement candidates section
    complement_text = ""
    if complement_candidates:
        complement_lines = []
        for wid in [p["work_id"] for p in papers_meta]:
            candidates = complement_candidates.get(wid, [])
            if not candidates:
                continue
            title = next((p.get("title", "Unknown") for p in papers_meta if p["work_id"] == wid), "Unknown")
            complement_lines.append(f"\nFor Paper [{wid}] \"{title}\":")
            complement_lines.append("Candidate complements (papers that address this paper's weaknesses):")
            for cand in candidates:
                cand_wid = cand.get("work_id", "?")
                cand_title = cand.get("title", "Unknown")
                cand_year = cand.get("year", "?")
                cand_cited = cand.get("cited_by_count", 0)
                cand_abstract = (cand.get("abstract") or "")[:200]
                complement_lines.append(
                    f"- [{cand_wid}] \"{cand_title}\" ({cand_year}, {cand_cited} citations): "
                    f"{cand_abstract}..."
                )
        complement_text = "\n".join(complement_lines) if complement_lines else ""

    work_id_list = [p["work_id"] for p in papers_meta]
    titles_map = {p["work_id"]: p.get("title", "Unknown") for p in papers_meta}

    # Special instructions for edge cases
    special_instructions = ""
    if has_survey:
        special_instructions += (
            "\n\n=== SPECIAL INSTRUCTIONS: SURVEY PAPER ===\n"
            "One or more papers in this comparison is a SURVEY/REVIEW paper (not a single methodology).\n"
            "For survey papers:\n"
            "- In 'paradigms', describe the survey's ORGANIZING FRAMEWORK (how it categorizes methods) "
            "as its paradigm\n"
            "- In 'handles_well', describe the survey's analytical contribution (taxonomy, comparison "
            "framework, identified trends)\n"
            "- In 'struggles_with', describe the survey's limitations (recency, coverage gaps, bias "
            "toward certain methods)\n"
            "- A survey does NOT have a single 'mechanism' — describe its analytical lens instead\n"
        )
    if low_overlap:
        special_instructions += (
            "\n\n=== SPECIAL INSTRUCTIONS: LOW METHODOLOGICAL OVERLAP ===\n"
            "These papers come from DIFFERENT sub-domains with limited methodological overlap.\n"
            "- In 'common_problem', identify the BROADEST shared challenge (e.g., 'learning from "
            "data' rather than a specific task)\n"
            "- Focus paradigm differences on their FUNDAMENTAL approach differences rather than "
            "task-specific details\n"
            "- In 'recommendation', emphasize that these serve DIFFERENT use cases — do not force "
            "a direct comparison of outcomes\n"
        )

    return (
        "=== METHODOLOGY PROFILES ===\n\n"
        + "\n\n".join(profiles)
        + source_text_section
        + "\n\n=== CITATION RELATIONSHIPS ===\n"
        + citations_text
        + ("\n\n=== COMPLEMENT CANDIDATES ===\n" + complement_text if complement_text else "")
        + special_instructions
        + "\n\n=== YOUR TASK ===\n"
        "You are helping researchers decide which paper(s) to use for their work. "
        "Produce a structured comparison with THREE sections:\n\n"
        "1. CONVERGENCE/DIVERGENCE ANALYSIS (paradigm-centered)\n"
        "   - Identify the COMMON PROBLEM all papers address\n"
        "   - Group papers by their PARADIGM (methodological approach)\n"
        "   - Explain how each paradigm tackles the problem differently\n\n"
        "2. STRENGTHS/WEAKNESSES MATRIX (paper-centered)\n"
        "   - For EACH paper: what it handles well, what it struggles with, key assumptions\n"
        "   - complemented_by: select the paper that BEST addresses each weakness from the "
        "COMPLEMENT CANDIDATES list. The complement does NOT have to be from the comparison set. "
        "Do NOT force a complement — if no candidate genuinely addresses the weakness, omit it. "
        "Explain the SPECIFIC mechanism by which the complement addresses the gap.\n\n"
        "3. RECOMMENDATION (decision-focused)\n"
        "   - Your quick take on the landscape\n"
        "   - Decision matrix: specific scenarios → which paper → why\n"
        "   - Can these methods be combined? How?\n\n"
        "=== FORMAT RULES ===\n"
        "1. All claims must be DEFINITIVE. No 'may', 'might', 'could', 'potentially'.\n"
        "2. Be SPECIFIC with technical mechanisms, not vague descriptions.\n"
        "3. Every limitation must explain the CAUSE (design choice) and CONSEQUENCE.\n"
        "4. For abstract_only papers, use domain knowledge to fill in details.\n"
        "5. Provide at least 3 decision scenarios covering different research contexts.\n"
        "6. SELF-REFERENCE RULE: When writing about a paper's OWN properties (in handles_well, "
        "struggles_with, assumptions, mechanism, evidence, cause, consequence), NEVER cite "
        "that paper's work_id AND never use phrases like 'the paper', 'the authors', "
        "'this work', 'the proposed method'. Describe the METHOD DIRECTLY.\n"
        "   WRONG: 'W12345 uses attention mechanisms to capture long-range dependencies'\n"
        "   WRONG: 'The paper proposes a novel framework for image recognition'\n"
        "   WRONG: 'The authors introduce a residual learning approach'\n"
        "   RIGHT: 'Identity shortcut connections bypass non-linear transformations, allowing "
        "gradients to flow directly through 152 layers without degradation'\n"
        "   Write about WHAT THE METHOD DOES, not about what the paper/authors do.\n"
        "7. CROSS-REFERENCE RULE: When referencing OTHER papers (in complemented_by.coverage, "
        "recommendation.why, combination_notes), ALWAYS use the work_id.\n"
        "   WRONG: 'DenseNet provides feature reuse'\n"
        "   RIGHT: '[W67890] provides direct feature reuse across all layers through dense concatenation'\n"
        "   WORK_ID FORMAT: OpenAlex IDs start with 'W' (e.g., W2194775991). Semantic Scholar IDs "
        "start with 'S2:' (e.g., S2:204e3073). ArXiv IDs start with 'AX:' (e.g., AX:1706.03762). "
        "NEVER use raw arxiv IDs like 'arxiv:1706.03762' or '1706.03762' — always prefix with 'AX:'.\n"
        "8. TECHNICAL DEPTH: Every mechanism, cause, consequence, and coverage description MUST "
        "include at least ONE of: (a) a named algorithm/technique, (b) a mathematical formulation, "
        "(c) a specific quantity/metric, (d) a concrete architectural detail.\n"
        "   SHALLOW (BANNED): 'provides a method for editing images'\n"
        "   DEEP (REQUIRED): 'uses DDPM inversion to find the noise map of the input image, then "
        "re-denoises with a new text prompt, applying a binary mask derived from cross-attention "
        "maps to preserve unedited regions'\n"
        "   SHALLOW (BANNED): 'increased computational requirements'\n"
        "   DEEP (REQUIRED): 'cascading through 3 upsampling stages (64→256→1024) requires ~3× the "
        "forward passes of a single-stage model, with each stage U-Net adding ~200M parameters'\n"
        "9. FACTUAL CLAIMS: Only cite specific numbers (percentages, parameter counts, FLOPs, "
        "dataset sizes, benchmark scores) if they appear in the paper text or profile above. "
        "If the exact number is not available, use qualitative comparisons instead.\n"
        "   WRONG (fabricated): 'achieves 3.57% top-5 error rate'\n"
        "   RIGHT (qualitative): 'achieves state-of-the-art top-5 error on ImageNet'\n"
        "   You MAY use well-known facts about widely-cited papers, but do NOT invent specific "
        "numbers for less well-known papers.\n"
        "10. NO CIRCULAR COMPLEMENTS: If Paper A lists Paper B in complemented_by, then Paper B "
        "MUST NOT list Paper A in its complemented_by. Circular references (A→B, B→A) are banned. "
        "Instead, find a DIFFERENT paper from the COMPLEMENT CANDIDATES for each paper's gaps. "
        "If no genuine complement exists, omit complemented_by entirely for that paper.\n"
        "11. GROUNDING IN SOURCE TEXT: Your mechanism, evidence, cause, consequence, and coverage "
        "descriptions MUST quote specific terms from the PAPER SOURCE TEXT and METHODOLOGY PROFILES. "
        "If the text says 'synchrosqueezed wavelet transforms (SWT)', write EXACTLY that — "
        "not 'wavelet-based processing'. If a profile's approach mentions 'cross-entropy loss "
        "with Adam optimizer (lr=1e-4)', write that — not 'standard training loss'. "
        "NEVER generalize terminology that appears in the source text or profiles.\n\n"
        "=== FULL WORKED EXAMPLE (study this level of depth) ===\n"
        "Given two GAN papers (pix2pix vs CycleGAN), here is what GOOD output looks like.\n"
        "Notice: every field quotes specific architectures, loss terms, and numbers.\n\n"
        '{\n'
        '  "convergence_divergence": {\n'
        '    "common_problem": {\n'
        '      "domain": "Image-to-Image Translation",\n'
        '      "challenge": "Translating images between visual domains while preserving structural content",\n'
        '      "why_hard": "The mapping between domains is ill-posed — infinitely many output images can correspond to a single input, and the generator must learn to produce only the plausible ones"\n'
        '    },\n'
        '    "paradigms": [\n'
        '      {\n'
        '        "name": "Paired Conditional Translation",\n'
        '        "papers": ["Wpix2pix"],\n'
        '        "mechanism": "Conditions a U-Net generator (8 encoder + 8 decoder layers with skip connections) on aligned input-output pairs, using PatchGAN 70×70 discriminator and combined loss L = L_cGAN(G,D) + λ·L1(G(x),y) where λ=100",\n'
        '        "philosophy": "Pixel-aligned supervision with L1 captures low-frequency structure while adversarial loss from PatchGAN captures high-frequency texture"\n'
        '      },\n'
        '      {\n'
        '        "name": "Unpaired Cycle-Consistent Translation",\n'
        '        "papers": ["Wcyclegan"],\n'
        '        "mechanism": "Learns two ResNet-9 generators (G: X→Y, F: Y→X) with cycle consistency loss F(G(X))≈X weighted by λ_cyc=10, using least-squares adversarial loss (LSGAN) instead of log-likelihood",\n'
        '        "philosophy": "Cycle consistency substitutes for paired supervision — if translation is meaningful, round-tripping should recover the original"\n'
        '      }\n'
        '    ],\n'
        '    "divergence_summary": "Both learn image-to-image mappings via adversarial training but diverge on supervision: pix2pix uses pixel-aligned pairs with L1+cGAN loss, while CycleGAN removes the pairing requirement by enforcing F(G(X))≈X cycle consistency with λ_cyc=10, trading reconstruction fidelity for domain flexibility."\n'
        '  },\n'
        '  "strengths_weaknesses_matrix": [\n'
        '    {\n'
        '      "work_id": "Wpix2pix",\n'
        '      "title": "Image-to-Image Translation with Conditional Adversarial Networks",\n'
        '      "handles_well": [\n'
        '        {\n'
        '          "capability": "High-fidelity paired translation",\n'
        '          "mechanism": "U-Net skip connections preserve spatial detail from encoder to decoder at each of 8 resolution levels, while L1 loss (λ=100) anchors output to ground truth",\n'
        '          "evidence": "Achieves 71.8% per-pixel accuracy (FCN-score) on Cityscapes labels→photo, outperforming L1-only (52.9%) and cGAN-only (60.3%)"\n'
        '        }\n'
        '      ],\n'
        '      "struggles_with": [\n'
        '        {\n'
        '          "limitation": "Requires pixel-aligned paired training data",\n'
        '          "cause": "L1 loss computes per-pixel difference between generated and ground-truth images, requiring exact spatial correspondence",\n'
        '          "consequence": "Cannot be applied to domains where paired data is unavailable (e.g., Monet paintings ↔ photographs), limiting applicability to ~5 datasets where alignment exists"\n'
        '        },\n'
        '        {\n'
        '          "limitation": "Tiling artifacts from PatchGAN receptive field",\n'
        '          "cause": "PatchGAN discriminator classifies independent 70×70 patches with no cross-patch communication",\n'
        '          "consequence": "Visible repetitive texture boundaries at 70-pixel intervals in generated images, particularly noticeable in uniform regions like sky or walls"\n'
        '        }\n'
        '      ],\n'
        '      "assumptions": [\n'
        '        {\n'
        '          "assumption": "Input and output images are spatially aligned at pixel level",\n'
        '          "if_violated": "L1 loss produces blurred outputs as it averages over misaligned pixels"\n'
        '        }\n'
        '      ],\n'
        '      "complemented_by": [\n'
        '        {\n'
        '          "other_work_id": "Wcyclegan",\n'
        '          "other_title": "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",\n'
        '          "other_year": 2017,\n'
        '          "coverage": "CycleGAN replaces L1 paired supervision with cycle consistency loss F(G(X))≈X (λ_cyc=10), enabling training on unpaired collections — directly addressing pix2pix\'s paired data requirement"\n'
        '        }\n'
        '      ]\n'
        '    },\n'
        '    {\n'
        '      "work_id": "Wcyclegan",\n'
        '      "title": "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",\n'
        '      "handles_well": [\n'
        '        {\n'
        '          "capability": "Translation without paired supervision",\n'
        '          "mechanism": "Cycle consistency loss F(G(X))≈X + G(F(Y))≈Y with λ_cyc=10 constrains the mapping space without requiring aligned pairs",\n'
        '          "evidence": "Produces visually plausible horse↔zebra, summer↔winter, photo↔Monet translations from fully unpaired collections of ~1000 images per domain"\n'
        '        }\n'
        '      ],\n'
        '      "struggles_with": [\n'
        '        {\n'
        '          "limitation": "Cannot handle geometric transformations",\n'
        '          "cause": "Cycle consistency F(G(X))≈X forces the generator to preserve spatial structure — any geometric change breaks the cycle",\n'
        '          "consequence": "Fails on tasks requiring shape changes (e.g., dog→cat), producing only texture/color transfer while preserving the source geometry"\n'
        '        }\n'
        '      ],\n'
        '      "assumptions": [\n'
        '        {\n'
        '          "assumption": "A bijective mapping exists between source and target domains",\n'
        '          "if_violated": "Many-to-one mappings (e.g., multiple cat breeds → one dog breed) cause mode collapse in the reverse generator F"\n'
        '        }\n'
        '      ],\n'
        '      "complemented_by": [\n'
        '        {\n'
        '          "other_work_id": "Wstargan",\n'
        '          "other_title": "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation",\n'
        '          "other_year": 2018,\n'
        '          "coverage": "StarGAN uses a single generator conditioned on domain labels with classification loss, handling N domains with one model instead of CycleGAN\'s O(N²) generator pairs"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ],\n'
        '  "recommendation": {\n'
        '    "summary": "Use pix2pix when aligned paired data exists (maps, segmentation masks, edges) for maximum fidelity via L1+cGAN. Use CycleGAN when only unpaired collections are available, accepting lower spatial precision for domain flexibility.",\n'
        '    "decision_matrix": [\n'
        '      {\n'
        '        "scenario": "Semantic segmentation label → photorealistic image with paired data",\n'
        '        "use": "Wpix2pix",\n'
        '        "why": "L1 loss (λ=100) on pixel-aligned pairs penalizes per-pixel deviation from ground truth, ensuring structural fidelity to the segmentation layout. The PatchGAN discriminator (70×70 receptive field) adds local texture realism without the blurring that full-image discriminators produce. Together they achieve 71.8% FCN-score vs CycleGAN\'s ~58%, because paired supervision lets the generator learn exact spatial correspondences."\n'
        '      },\n'
        '      {\n'
        '        "scenario": "Artistic style transfer between unpaired image collections",\n'
        '        "use": "Wcyclegan",\n'
        '        "why": "Cycle consistency loss (F(G(X))≈X with λ_cyc=10) enables learning from unpaired collections where no pixel correspondence exists — e.g., ~1000 Monet paintings + ~1000 photographs with no paired samples. By enforcing round-trip reconstruction, the generator learns domain-specific style transfer without requiring aligned training pairs that pix2pix demands. This makes it the only viable approach when collecting paired data is impractical or impossible."\n'
        '      },\n'
        '      {\n'
        '        "scenario": "Domain adaptation for autonomous driving (sim→real)",\n'
        '        "use": "Wcyclegan",\n'
        '        "why": "Simulated and real driving images have no pixel alignment, ruling out paired approaches like pix2pix. CycleGAN\'s cycle consistency loss preserves scene layout (road geometry, lane markings, vehicle positions) while transferring visual appearance from rendered to photorealistic domain using the ResNet-9 generator\'s instance normalization. This enables training on existing simulator outputs without expensive manual annotation of real-world driving scenes."\n'
        '      }\n'
        '    ],\n'
        '    "can_combine": true,\n'
        '    "combination_notes": "When sparse paired examples exist alongside large unpaired collections, use CycleGAN\'s cycle loss for the bulk of training and add pix2pix\'s L1 loss on the paired subset as an auxiliary objective."\n'
        '  }\n'
        '}\n\n'
        "NOTICE how every field in the example above contains:\n"
        "- Specific architecture names (U-Net, ResNet-9, PatchGAN 70×70)\n"
        "- Exact loss formulations (L = L_cGAN + λ·L1 where λ=100, F(G(X))≈X with λ_cyc=10)\n"
        "- Concrete numbers (71.8% FCN-score, 8 encoder layers, ~1000 images)\n"
        "- Causal chains in cause→consequence (not just 'increases cost')\n"
        "- RECOMMENDATION WHY fields must be 2-3 sentences explaining the FULL causal chain:\n"
        "  (1) Name the specific mechanism/technique from the paper\n"
        "  (2) Explain HOW that mechanism addresses this particular scenario\n"
        "  (3) Provide evidence: a metric, benchmark result, or architectural detail that confirms it\n"
        "  NOT: 'Achieves state-of-the-art FID scores on image generation tasks'\n"
        "  NOT: 'Provides good results for this task'\n"
        "  NOT: 'By providing robust feature extractors that can be fine-tuned' (VAGUE — what features? how?)\n"
        "  YES: 'L1 loss (λ=100) on pixel-aligned pairs ensures structural fidelity to the segmentation "
        "layout by penalizing per-pixel deviation from ground truth. PatchGAN (70×70 receptive field) adds "
        "local texture realism without blurring. Together they achieve 71.8% FCN-score vs CycleGAN's ~58%.'\n"
        "  The BY/USING clause must name a SPECIFIC named technique with dimensions or parameters — "
        "never 'by providing robust/better/efficient X'. Name the actual component.\n"
        "  Every 'why' must name the mechanism, explain how it helps, and give evidence.\n"
        "Your output MUST match this level of specificity. Generic descriptions will be rejected.\n\n"
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
            '"complemented_by": [{"other_work_id": "W...", "other_title": "...", "other_year": 2020, '
            '"coverage": "SPECIFIC mechanism: explain exactly how this paper addresses the gap"}]}'
            for wid in work_id_list
        )
        + "\n  ],\n"
        '  "recommendation": {\n'
        '    "summary": "Your quick take on which paper to use and when",\n'
        '    "decision_matrix": [\n'
        '      {"scenario": "Specific use case", "use": "' + work_id_list[0] + '", '
        '"why": "..."},\n'
        '      {"scenario": "Different use case", "use": "' + (work_id_list[1] if len(work_id_list) > 1 else work_id_list[0]) + '", '
        '"why": "..."},\n'
        '      {"scenario": "Third use case", "use": "...", '
        '"why": "..."}\n'
        '    ],\n'
        '    "can_combine": true/false,\n'
        '    "combination_notes": "Specific technique for combining: name the components from each paper"\n'
        '  }\n'
        "}"
    )


def _synthesize(
    papers_meta: list[Dict[str, Any]],
    fingerprints: Dict[str, Dict[str, Any]],
    contents: list[PaperContent],
    lineage: CitationLineage,
    complement_candidates: Optional[Dict[str, list[Dict[str, Any]]]] = None,
    has_survey: bool = False,
    low_overlap: bool = False,
) -> Dict[str, Any]:
    """Run the synthesis LLM call."""
    prompt = _build_synthesis_prompt(
        papers_meta, fingerprints, contents, lineage, complement_candidates,
        has_survey=has_survey, low_overlap=low_overlap,
    )
    work_ids = [p["work_id"] for p in papers_meta]

    # Build set of all valid complement work_ids (input + candidates)
    valid_complement_wids = set(work_ids)
    if complement_candidates:
        for cands in complement_candidates.values():
            for cand in cands:
                cand_wid = cand.get("work_id", "")
                if cand_wid:
                    valid_complement_wids.add(cand_wid)

    relaxed = has_survey or low_overlap
    result = _call_llm(
        SYNTHESIS_SYSTEM, prompt,
        validator=_validate_synthesis,
        validator_args=(work_ids, valid_complement_wids, relaxed, fingerprints),
        relaxed_validator_args=(work_ids, valid_complement_wids, True, fingerprints),
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
    rank_job_id: Optional[str] = None,
) -> MethodologyComparisonResponse:
    """
    Shared comparison pipeline (post-validation).

    Steps:
    1. Check comparison cache
    2. Stage 1A + 1B: Fetch content + build lineage (parallel)
    3. Stage 2: Extract fingerprints (LLM call 1)
    3.5. Find complement candidates (DB + external APIs)
    4. Stage 3: Synthesize comparison (LLM call 2)
    5. Scrub self-references
    6. Cache and return
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

    # 3a. Detect paper types and overlap
    paper_types = {wid: _detect_paper_type(fingerprints.get(wid, {})) for wid in work_ids}
    has_survey = any(t == "survey" for t in paper_types.values())
    if has_survey:
        logger.info(f"Survey paper detected: {[w for w, t in paper_types.items() if t == 'survey']}")

    min_overlap = 1.0
    for i, wid1 in enumerate(work_ids):
        for wid2 in work_ids[i + 1:]:
            overlap = _compute_paper_overlap(
                fingerprints.get(wid1, {}), fingerprints.get(wid2, {})
            )
            min_overlap = min(min_overlap, overlap)
    low_overlap = min_overlap < 0.10
    if low_overlap:
        logger.info(f"Low methodological overlap detected (Jaccard={min_overlap:.3f})")

    # 3.5. Find complement candidates
    complement_candidates = _find_complement_candidates(
        conn, work_ids, fingerprints, papers_meta, map_id, rank_job_id
    )

    # 4. Stage 3: Synthesize comparison
    synthesis = _synthesize(
        papers_meta, fingerprints, contents, lineage, complement_candidates,
        has_survey=has_survey, low_overlap=low_overlap,
    )

    # 5. Normalize work_ids and scrub self-references
    synthesis = _normalize_work_ids(synthesis)
    synthesis = _scrub_self_references(synthesis)

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
                    other_title=c.get("other_title"),
                    other_year=c.get("other_year"),
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
            conn, tenant_id, work_ids, papers_meta_map,
            map_id=None, rank_job_id=rank_job_id
        )
