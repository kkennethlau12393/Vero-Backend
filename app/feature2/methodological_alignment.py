"""
Quality filters for ranked results — data quality, review override,
and domain boundary enforcement.

Extracted from the former categorization pipeline. These filters run
before focus filtering to remove noise from the flat ranked list.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Programmatic boundary enforcement (deterministic safety net over LLM)
# ---------------------------------------------------------------------------
# The LLM unreliably enforces domain/modality/method boundaries. These
# programmatic checks catch papers where the title clearly indicates a
# different domain than the query, regardless of what the LLM scored.
#
# Three independent categories of checks:
# 1. ML modality groups (NLP vs vision vs audio etc.)
# 2. Scientific system groups (bacterial vs cancer vs parasitology)
# 3. Generative method groups (GAN vs diffusion vs VAE etc.)
#
# Each check only fires when BOTH query and title have detectable groups
# within the same category AND they don't overlap. If either side has no
# detectable group, no filtering occurs (safe default).

_ML_MODALITY_GROUPS = {
    "nlp": ["language", " nlp", "text ", "linguistic", "translation", "sentiment",
            "named entity", "question answering", "reading comprehension",
            "summarization", "dialogue"],
    "vision": ["image", "vision", "visual", "segmentation", "object detection",
               "pixel", "video", "face recognition", "pose estimation"],
    "medical_imaging": ["medical image", "pathology", "radiology", "x-ray",
                        "ct scan", "mri ", "ultrasound", "histopathology"],
    "timeseries": ["time-series", "time series", "forecasting", "temporal prediction"],
    "audio": ["audio", "speech recognition", "sound", "acoustic", "music"],
    "graph": ["graph neural", "graph attention", "graph convolution",
              "node classification", "link prediction"],
    "robotics": ["robot", "manipulation", "grasping", "locomotion", "navigation",
                 "humanoid", "drone"],
    "genomics": ["dna", "genomic", "genome", "deoxyribonucleic", "nucleotide",
                 "amino acid", "protein fold", "gene sequence"],
}

# Scientific system groups - catches cross-system confusion where the same
# phenomenon name (e.g., "drug resistance") appears in different biological systems.
_SCIENTIFIC_SYSTEM_GROUPS = {
    "bacterial": ["antibiotic", "antimicrobial", " bacteria", "bacterial",
                  "pseudomonas", "staphylococc", "streptococc", "enterococc",
                  "escherichia", "klebsiella", "acinetobacter", "biofilm",
                  "beta-lactam", "efflux pump"],
    "oncology": [" tumor", " tumour", " cancer", "carcinoma", "melanoma",
                 "leukemia", "leukaemia", "lymphoma", "neoplasm", " metasta",
                 "oncolog", "chemotherap", "anti-tumor", "antitumor"],
    "parasitology": ["leishmani", " malaria", "plasmodium", " parasite",
                     "parasit", "helminth", "protozoa", "trypanosoma",
                     "schistosom"],
}

# Generative method groups - catches method boundary violations where a paper
# uses a different generative method than what the query specifies.
_GENERATIVE_METHOD_GROUPS = {
    "gan": ["generative adversarial", " gan ", " gans ", "adversarial network",
            " dcgan", "cyclegan", "stylegan", "progan", "biggan",
            "wasserstein gan", " wgan"],
    "diffusion": ["diffusion model", "denoising diffusion", " ddpm", " ddim",
                  "score-based generative", "latent diffusion",
                  "stable diffusion"],
    "autoregressive_gen": ["pixelcnn", "pixelrnn"],
    "vae": ["variational autoencoder", " vae ", " vaes "],
}

# All group categories to check (order doesn't matter, all are independent)
_ALL_CONFLICT_GROUPS = [
    _ML_MODALITY_GROUPS,
    _SCIENTIFIC_SYSTEM_GROUPS,
    _GENERATIVE_METHOD_GROUPS,
]


# ---------------------------------------------------------------------------
# Title-based review/survey detection
# ---------------------------------------------------------------------------
# Deterministic title-pattern detection for papers that are clearly reviews,
# surveys, or meta-analyses. The LLM may miss these if the abstract doesn't
# emphasize the review nature. Title patterns are highly reliable.
_REVIEW_TITLE_PATTERNS = [
    "a survey of ",
    "a survey on ",
    "survey of ",
    "survey on ",
    "a review of ",
    "a review on ",
    "review of ",
    "review on ",
    "a comprehensive survey",
    "a comprehensive review",
    "a systematic review",
    "systematic review of",
    "systematic review on",
    "meta-analysis of",
    "meta-analysis on",
    "a meta-analysis",
    ": a survey",
    ": a review",
    ": a comprehensive",
    ": a systematic",
    "literature review",
    "state-of-the-art review",
    "state of the art review",
    "tutorial on ",
    "a tutorial on ",
    ": a tutorial",
]


def is_review_by_title(title: str) -> bool:
    """Detect review/survey papers by title patterns."""
    if not title:
        return False
    title_lower = f" {title.lower().strip()} "
    return any(pattern in title_lower for pattern in _REVIEW_TITLE_PATTERNS)


def _detect_groups(text: str, groups: dict) -> set:
    """Detect which groups from a category a text belongs to."""
    text_lower = f" {text.lower()} "
    found = set()
    for group_name, indicators in groups.items():
        if any(ind in text_lower for ind in indicators):
            found.add(group_name)
    return found


def _check_group_conflict(query_text: str, title: str, groups: dict) -> bool:
    """Return True if query and title belong to DIFFERENT groups within a category.

    Only fires when both have detectable groups and they don't overlap.
    """
    query_groups = _detect_groups(query_text, groups)
    if not query_groups:
        return False
    title_groups = _detect_groups(title, groups)
    if not title_groups:
        return False
    return len(query_groups & title_groups) == 0


def check_all_domain_conflicts(query_text: str, title: str) -> bool:
    """Run all programmatic conflict checks. Returns True if ANY conflict detected."""
    return any(
        _check_group_conflict(query_text, title, groups)
        for groups in _ALL_CONFLICT_GROUPS
    )


# Minimum LLM relevance for domain boundary check trigger
_MIN_RELEVANCE_FOR_BOUNDARY_CHECK = 0.40


def apply_quality_filters(
    ranked_items: List[Dict],
    llm_scores: Dict[str, Dict[str, Any]],
    query_text: Optional[str] = None,
) -> List[Dict]:
    """Apply data quality and domain boundary filters to ranked items.

    Extracted from partition_results_by_category — these filters must run
    before focus filtering to remove noise from the flat list.

    Filters applied:
    1. Data quality: missing title/year, future year, current year with 0 citations
    2. Title-based review override: override LLM paper_type if title matches
    3. Domain boundary: filter papers where title indicates different domain
    """
    current_year = date.today().year
    result = []
    stats = {"bad_data": 0, "domain_conflict": 0, "review_override": 0}

    for item in ranked_items:
        preview = item.get("preview", {})
        title = preview.get("title", "")
        year = preview.get("year")
        citations = preview.get("cited_by_count", 0) or 0
        work_id = item.get("work_id")

        # Filter 1: Data quality
        if not title or year is None:
            stats["bad_data"] += 1
            continue
        if year > current_year:
            stats["bad_data"] += 1
            continue
        if year >= current_year and citations == 0:
            stats["bad_data"] += 1
            continue

        # Filter 2: Title-based review override
        llm_data = llm_scores.get(work_id, {})
        paper_type = llm_data.get("paper_type", "other")
        if paper_type != "review" and is_review_by_title(title):
            breakdown = item.get("breakdown", {})
            breakdown["paper_type"] = "review"
            item["breakdown"] = breakdown
            stats["review_override"] += 1

        # Filter 3: Domain boundary check
        relevance = llm_data.get("score", 0.0)
        if query_text and relevance >= _MIN_RELEVANCE_FOR_BOUNDARY_CHECK:
            if check_all_domain_conflicts(query_text, title):
                logger.debug(
                    f"Domain conflict: '{title[:60]}' (llm={relevance}) "
                    f"filtered for query '{query_text[:40]}'"
                )
                stats["domain_conflict"] += 1
                continue

        result.append(item)

    total_filtered = stats["bad_data"] + stats["domain_conflict"]
    if total_filtered > 0:
        logger.info(
            f"Quality filter: {stats['bad_data']} bad data, "
            f"{stats['domain_conflict']} domain conflict, "
            f"{stats['review_override']} review overrides "
            f"({len(ranked_items)} → {len(result)} papers)"
        )

    return result
