"""
Paper categorization using LLM paper types + citation-based heuristics.

The LLM classifies paper TYPE (foundational/methodology/review/application/etc).
Python determines OUTPUT CATEGORY using type + citations + year.

4 OUTPUT CATEGORIES:
- foundational: High-citation seminal papers that introduced key concepts
- methodology: Methods, algorithms, tools, frameworks
- reviews: Surveys, reviews, meta-analyses, tutorials
- applications: Clinical trials, case studies, real-world implementations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
# Tool/software paper detection
# ---------------------------------------------------------------------------
# Papers primarily about software tools, databases, web servers, or analysis
# platforms are capped at "methodology" and cannot be "foundational". They are
# useful instruments used in the field, not the intellectual breakthroughs.
_TOOL_PAPER_INDICATORS = [
    "web server", "web portal", "web tool", "web-based tool",
    "web service for", "online server", "online tool", "online resource",
    "database:", "updated database", "database in 20",
    " server for ", " server:",
    "toolkit", "toolbox", "software tool", "software package",
    "r package", "python package", "bioconductor",
    "analysis pipeline", "computational pipeline",
    "comprehensive integration",
    "user-friendly",
    "freely available at",
]


def _is_tool_paper(title: str) -> bool:
    """Detect papers that are primarily software tools, databases, or platforms."""
    title_lower = f" {title.lower()} "
    return any(ind in title_lower for ind in _TOOL_PAPER_INDICATORS)


# ---------------------------------------------------------------------------
# Textbook detection
# ---------------------------------------------------------------------------
# General textbooks get their own category rather than mixing with research.
_TEXTBOOK_TITLE_PREFIXES = [
    "introduction to ",
    "an introduction to ",
    "textbook of ",
    "a textbook of ",
]

_TEXTBOOK_TITLE_CONTAINS = [
    " handbook of ",
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


def _is_textbook(title: str) -> bool:
    """Detect general textbooks by title patterns."""
    title_lower = title.lower().strip()
    if any(title_lower.startswith(prefix) for prefix in _TEXTBOOK_TITLE_PREFIXES):
        return True
    title_padded = f" {title_lower} "
    if any(phrase in title_padded for phrase in _TEXTBOOK_TITLE_CONTAINS):
        return True
    return False


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


# Minimum LLM relevance score - papers below this are filtered out in categorization
# With continuous 0-10 scoring (normalized to 0-1): 0.40 = 4/10 = "tangentially related"
# RRF naturally demotes low-quality papers, so this threshold can be lower
MIN_LLM_RELEVANCE_THRESHOLD = 0.40

# Category limits - 6 categories (5 typed + 1 overflow)
DEFAULT_CATEGORY_LIMITS = {
    "foundational": 10,         # +2: at-cap 48% of runs
    "methodology": 16,          # +4: at-cap 96% — biggest gain
    "reviews": 10,              # +2: modest bump
    "applications": 14,         # +4: at-cap 48%, broad queries fill it
    "textbooks": 2,             # -3: never fills (avg 0.1/5)
    "additional_relevant": 10,  # overflow for quality papers that fail citation thresholds
}

@dataclass
class DynamicThresholds:
    """Thresholds computed from the field's distribution."""
    legendary: int           # Top ~15% - field-defining papers
    foundational_min: int    # Top ~25% - high-impact papers
    methodology_min: int     # Top ~60% - decent citations
    review_min: int          # Top ~70% - validated reviews
    application_min: int     # Top ~85% - some validation
    min_foundational_age: int  # Median age - papers older than this can be foundational


def compute_dynamic_thresholds(
    all_citations: List[int],
    all_ages: Optional[List[int]] = None,
) -> DynamicThresholds:
    """Compute thresholds based on the field's distribution.

    Uses percentiles so niche fields have lower thresholds than high-citation fields.
    Age threshold is based on median age of papers in the pool.
    No hardcoded minimums - purely relative to the candidate pool.
    """
    if len(all_citations) < 5:
        # Too few papers - use zeros (everything passes)
        return DynamicThresholds(
            legendary=0,
            foundational_min=0,
            methodology_min=0,
            review_min=0,
            application_min=0,
            min_foundational_age=0,
        )

    sorted_cites = sorted(all_citations, reverse=True)
    n = len(sorted_cites)

    # Get citation values at percentile positions
    def get_percentile(pct: float) -> int:
        idx = min(int(n * pct), n - 1)
        return sorted_cites[idx]

    # All thresholds are purely percentile-based
    legendary = get_percentile(0.15)           # Top 15%
    foundational_min = get_percentile(0.25)    # Top 25%
    methodology_min = get_percentile(0.60)     # Top 60%
    review_min = get_percentile(0.70)          # Top 70%
    application_min = get_percentile(0.85)     # Top 85%

    # Compute age threshold: 25th percentile (older than 75% of papers)
    # Using 25th instead of median prevents mature fields from having too-high thresholds
    if all_ages and len(all_ages) >= 5:
        sorted_ages = sorted(all_ages)
        age_25th = sorted_ages[len(sorted_ages) // 4]
        min_foundational_age = max(2, age_25th)
    else:
        min_foundational_age = 2

    thresholds = DynamicThresholds(
        legendary=legendary,
        foundational_min=foundational_min,
        methodology_min=methodology_min,
        review_min=review_min,
        application_min=application_min,
        min_foundational_age=min_foundational_age,
    )

    logger.info(
        f"Dynamic thresholds: legendary={legendary}, foundational={foundational_min}, "
        f"methodology={methodology_min}, review={review_min}, application={application_min}"
    )

    return thresholds


def get_min_citations_for_age(age_years: int) -> int:
    """Get minimum citation threshold based on paper age.

    Older papers have had more time to accumulate citations, so we expect more.
    This filters out old obscure papers while allowing newer impactful ones.
    """
    if age_years >= 15:
        return 100  # 15+ years old: should have 100+ cites
    elif age_years >= 10:
        return 50   # 10-14 years: should have 50+ cites
    elif age_years >= 5:
        return 25   # 5-9 years: should have 25+ cites
    elif age_years >= 2:
        return 10   # 2-4 years: should have 10+ cites
    else:
        return 0    # 0-1 years: new papers, no floor


def determine_output_category(
    paper_type: str,
    citations: int,
    year: int,
    current_year: int,
    llm_relevance: float,
    thresholds: Optional[DynamicThresholds] = None,
) -> Optional[str]:
    """Determine output category using paper type + citations + year.

    Returns None if paper should be filtered out entirely.

    Logic:
    1. foundational: Top 15% citations + older than median age
    2. reviews: LLM-typed reviews
    3. applications: LLM-typed applications
    4. methodology: Everything else with decent citations
    """
    age_years = current_year - year if year else 0

    # Use dynamic thresholds (should always be provided, but default to zeros if not)
    if thresholds is None:
        thresholds = DynamicThresholds(
            legendary=0,
            foundational_min=0,
            methodology_min=0,
            review_min=0,
            application_min=0,
            min_foundational_age=2,
        )

    # === CATEGORY ROUTING ===

    is_old_enough = age_years >= thresholds.min_foundational_age
    is_highly_cited = citations >= thresholds.legendary  # Top 15%

    # Priority 1: Foundational - top 15% citations + old enough
    # With continuous 0-10 scoring: 0.65 = 6.5/10 = between "useful context" and
    # "directly addresses". Papers scoring 5/10 ("useful context, not primarily about
    # the query") should NOT be labeled foundational — that lets XGBoost/TensorFlow
    # into foundational for specialized queries just because they're highly cited.
    FOUNDATIONAL_MIN_RELEVANCE = 0.65
    if is_highly_cited and is_old_enough and llm_relevance >= FOUNDATIONAL_MIN_RELEVANCE:
        return "foundational"

    # For non-foundational papers, apply standard LLM relevance filter
    if llm_relevance < MIN_LLM_RELEVANCE_THRESHOLD:
        return None

    # Priority 2: Reviews - LLM-typed reviews with enough citations
    if paper_type == "review":
        if citations >= thresholds.review_min:
            return "reviews"
        else:
            return None

    # Priority 3: Applications - LLM-typed applications with validation
    if paper_type == "application":
        is_current_year = (year == current_year)
        if citations >= thresholds.application_min or (is_current_year and citations >= 0):
            return "applications"
        else:
            return None

    # Priority 4: Methodology - methods, tools, algorithms
    if paper_type in ("methodology", "foundational", "theoretical", "other"):
        # Use dynamic threshold, with recency exceptions
        if citations >= thresholds.methodology_min:
            return "methodology"
        elif age_years <= 2 and citations >= 3:
            return "methodology"
        elif age_years <= 4 and citations >= 8:
            return "methodology"
        else:
            return None

    # Fallback: methodology
    return "methodology"


def partition_results_by_category(
    ranked_items: List[Dict],
    llm_scores: Dict[str, Dict[str, Any]],
    category_limits: Optional[Dict[str, int]] = None,
    query_text: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Partition ranked results into 4 categories using LLM types + citations.

    Args:
        ranked_items: List of ranked result dicts with 'work_id' key
        llm_scores: Dict mapping work_id to {"score": float, "paper_type": str}
        category_limits: Optional dict overriding default limits
        query_text: Optional query text for programmatic modality boundary check

    Returns:
        Dict with 4 category keys, each containing a list of items up to the limit.
    """
    limits = {**DEFAULT_CATEGORY_LIMITS, **(category_limits or {})}
    current_year = date.today().year

    # Compute dynamic thresholds based on citation and age distributions
    all_citations = []
    all_ages = []
    for item in ranked_items:
        preview = item.get("preview", {})
        citations = preview.get("cited_by_count", 0) or 0
        year = preview.get("year")
        all_citations.append(citations)
        if year and year <= current_year:
            all_ages.append(current_year - year)

    thresholds = compute_dynamic_thresholds(all_citations, all_ages)

    # Collect ALL papers per category (no limits yet)
    buckets: Dict[str, List[Dict]] = {
        "foundational": [],
        "methodology": [],
        "reviews": [],
        "applications": [],
        "textbooks": [],
        "additional_relevant": [],
    }

    filtered_stats = {"low_relevance": 0, "low_citations": 0, "bad_data": 0,
                      "domain_conflict": 0, "tool_paper_capped": 0,
                      "textbook": 0}

    for item in ranked_items:
        work_id = item.get("work_id")
        preview = item.get("preview", {})
        title = preview.get("title", "")
        year = preview.get("year")
        citations = preview.get("cited_by_count", 0) or 0

        # === DATA QUALITY FILTERS ===
        if not title or year is None:
            filtered_stats["bad_data"] += 1
            continue
        if year > current_year:
            filtered_stats["bad_data"] += 1
            continue
        if year >= current_year and citations == 0:
            filtered_stats["bad_data"] += 1
            continue

        # Get LLM data
        llm_data = llm_scores.get(work_id, {})
        relevance = llm_data.get("score", 0.0)
        paper_type = llm_data.get("paper_type", "other")

        # Title-based review detection overrides LLM paper_type
        # Catches reviews/surveys the LLM missed (title is highly reliable)
        if paper_type != "review" and is_review_by_title(title):
            paper_type = "review"

        # === TEXTBOOK CHECK ===
        # General textbooks get their own category, not mixed with research.
        if _is_textbook(title):
            if relevance >= MIN_LLM_RELEVANCE_THRESHOLD:
                buckets["textbooks"].append(item)
                filtered_stats["textbook"] += 1
            else:
                filtered_stats["low_relevance"] += 1
            continue

        # === DOMAIN BOUNDARY CHECK ===
        # Deterministic safety net: catch papers where the LLM gave HIGH but
        # the title clearly indicates a different domain/modality/method.
        # Checks: ML modalities, scientific systems, generative methods.
        if query_text and relevance >= MIN_LLM_RELEVANCE_THRESHOLD:
            if check_all_domain_conflicts(query_text, title):
                logger.debug(
                    f"Domain conflict: '{title[:60]}' (llm={relevance}) "
                    f"filtered for query '{query_text[:40]}'"
                )
                filtered_stats["domain_conflict"] += 1
                continue

        # === DETERMINE CATEGORY ===
        category = determine_output_category(
            paper_type=paper_type,
            citations=citations,
            year=year,
            current_year=current_year,
            llm_relevance=relevance,
            thresholds=thresholds,
        )

        if category is None:
            if relevance < MIN_LLM_RELEVANCE_THRESHOLD:
                filtered_stats["low_relevance"] += 1
            else:
                # Paper passed LLM relevance + RRF but failed citation thresholds.
                # Route to overflow instead of discarding — these are quality papers
                # validated by 4 independent signals (BM25, TF-IDF, impact, LLM).
                buckets["additional_relevant"].append(item)
                filtered_stats["overflow"] = filtered_stats.get("overflow", 0) + 1
            continue

        # === TOOL PAPER CAP ===
        # Software tools, databases, web servers → methodology at most.
        # They are useful instruments, not the intellectual breakthroughs.
        if category == "foundational" and _is_tool_paper(title):
            logger.debug(
                f"Tool paper capped: '{title[:60]}' → methodology"
            )
            category = "methodology"
            filtered_stats["tool_paper_capped"] += 1

        buckets[category].append(item)

    # Sort each bucket and truncate to limits
    # ALL categories now sort by RRF score (item["score"]) — the combined relevance
    # from 4 rankers (BM25, TF-IDF, citation impact, LLM). This ensures papers that
    # are consistently good across all signals rank higher than papers with a single
    # strength (e.g., high citations but low topical relevance).
    for category in buckets:
        if buckets[category]:
            # Sort by RRF score (already computed and stored in item["score"])
            # Papers with higher combined relevance across all 4 rankers rank first.
            # This replaces the previous citation-only sorting which promoted off-topic
            # highly-cited papers (e.g., CIFAR-10 for diffusion, MAML for policy gradient).
            buckets[category].sort(
                key=lambda x: x.get("score", 0),
                reverse=True,
            )
        buckets[category] = buckets[category][:limits.get(category, 10)]

    # Log filtering stats
    total_filtered = sum(filtered_stats.values())
    if total_filtered > 0:
        extras = []
        if filtered_stats.get("domain_conflict", 0) > 0:
            extras.append(f"{filtered_stats['domain_conflict']} domain conflict")
        if filtered_stats.get("tool_paper_capped", 0) > 0:
            extras.append(f"{filtered_stats['tool_paper_capped']} tool→methodology")
        if filtered_stats.get("textbook", 0) > 0:
            extras.append(f"{filtered_stats['textbook']} textbooks")
        if filtered_stats.get("overflow", 0) > 0:
            extras.append(f"{filtered_stats['overflow']} → additional_relevant")
        extras_str = (", " + ", ".join(extras)) if extras else ""
        logger.info(
            f"Filtered {total_filtered} papers: "
            f"{filtered_stats['low_relevance']} low relevance, "
            f"{filtered_stats['low_citations']} low citations, "
            f"{filtered_stats['bad_data']} bad data{extras_str}"
        )

    counts = {k: len(v) for k, v in buckets.items()}
    logger.info(f"Partitioned by category: {counts}")

    return buckets
