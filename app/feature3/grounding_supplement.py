"""
Grounding paper supplement for Feature 3.

This module ensures novelty assessments have sufficient grounding papers
of BOTH types: reference-type (methodological) and landmark-type (field context).

Target distribution:
- At least 2-3 reference-type papers (direct methodology comparison)
- At least 2-3 landmark-type papers (historical field context)
- Total: 5-7 grounding papers

Note: Paper caching is handled by paper_cache.py for sharing between
grounding_supplement (novelty) and node_timeline features.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.engine import Connection

from app.feature3.json_utils import extract_json_from_llm_response_with_repair
from app.feature3.paper_identity import decode_openalex_abstract
from app.feature3.methodology import get_methodology, is_methodology_mismatch
from app.feature3.paper_cache import (
    get_landmarks,
    cache_landmarks,
)
from app.feature3.abstract_enrichment import (
    S2_RATE_LIMITER,
    ARXIV_RATE_LIMITER,
    API_TIMEOUT as EXTERNAL_API_TIMEOUT,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Groq Llama 4 Maverick - fast and high quality
MODEL_VERSION = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Grounding requirements
MIN_REFS = 2          # Minimum reference-type papers
MIN_LANDMARKS = 3     # Minimum landmark-type papers (3 for robust field context)
MIN_TOTAL = 5         # Minimum total grounding papers
MAX_TOTAL = 7         # Maximum total grounding papers

# API settings
OPENALEX_TIMEOUT = 15
MAX_RETRIES = 2
RETRY_BACKOFF = 0.5


# ============================================================================
# LLM Prompts for Different Paper Types
# ============================================================================

SUGGEST_REFS_PROMPT = """You are an expert academic researcher. Suggest papers that are METHODOLOGICALLY SIMILAR to the target paper - papers that use similar techniques, approaches, or solve similar problems.

Target Paper:
Title: {title}
Abstract: {abstract}
Year: {year}

Papers I already have:
{existing_papers}

Suggest {num_needed} papers that:
1. Use SIMILAR METHODS or techniques as the target paper
2. Were published BEFORE {year}
3. Are directly comparable in methodology
4. Are NOT already in my list
5. MUST be from the SAME FIELD - if target is a statistics paper, suggest statistics papers; if biology, suggest biology papers

CRITICAL: Do NOT suggest papers from different domains! A statistics paper should NEVER cite chemistry or physics papers for methodological comparison.

Return a JSON array:
[
    {{
        "title": "Exact paper title",
        "authors": "First Author et al.",
        "year": 2015,
        "why_relevant": "Uses similar [specific technique] for [specific task]"
    }}
]

Focus on methodological similarity, not just topic similarity."""


SUGGEST_LANDMARKS_PROMPT = """You are an expert academic researcher. Suggest SEMINAL/FOUNDATIONAL papers that use the SAME METHODOLOGY as the target paper.

Target Paper:
Title: {title}
Abstract: {abstract}
Year: {year}
Field/Topic: {field}

Papers I already have:
{existing_papers}

Suggest {num_needed} LANDMARK papers that:
1. Are highly cited (1000+ citations typically)
2. Use the SAME METHODOLOGY or TECHNIQUE as the target paper
3. Were published BEFORE {year}
4. Are NOT already in my list

Return a JSON array:
[
    {{
        "title": "Exact paper title",
        "authors": "First Author et al.",
        "year": 2012,
        "why_relevant": "Introduced [specific technique] that this paper builds upon"
    }}
]

Focus on papers that established the SPECIFIC TECHNIQUE used in the target paper."""


SUGGEST_PIONEERING_CONTEXT_PROMPT = """You are an expert academic researcher. This target paper appears to be a PIONEERING/FOUNDATIONAL work itself. Suggest papers that provide HISTORICAL CONTEXT for understanding its significance.

Target Pioneering Paper:
Title: {title}
Abstract: {abstract}
Year: {year}
Field/Topic: {field}

Papers I already have:
{existing_papers}

Suggest {num_needed} papers that provide CONTEXT for this pioneering work:
1. Papers published BEFORE or AROUND {year} in the SAME FIELD that show what existed prior
2. Contemporaneous works addressing similar problems (to show the intellectual climate)
3. Papers that cite this work heavily (showing its influence, even if published after)
4. Earlier foundational works in ADJACENT fields that may have influenced it

IMPORTANT:
- Papers must be from the SAME or closely related field (e.g., psychology, cognitive science for a psychology paper)
- Do NOT suggest papers from unrelated domains
- Focus on papers that help explain WHY this work was groundbreaking

Return a JSON array:
[
    {{
        "title": "Exact paper title",
        "authors": "First Author et al.",
        "year": 1970,
        "why_relevant": "Shows the state of [field] before this work / Contemporaneous work addressing similar questions / Key work influenced by this pioneering paper"
    }}
]

Focus on papers from the SAME DOMAIN that illuminate the historical significance."""


# ============================================================================
# Core Functions
# ============================================================================

def assess_grounding_needs(
    refs: List[Dict[str, Any]],
    landmarks: List[Dict[str, Any]],
) -> Tuple[int, int, str]:
    """
    Assess what additional grounding papers are needed.

    ALWAYS requires at least MIN_LANDMARKS (3) landmarks for proper field context,
    regardless of total count.

    Returns:
        (refs_needed, landmarks_needed, reason)
    """
    num_refs = len(refs)
    num_landmarks = len(landmarks)
    total = num_refs + num_landmarks

    # Even when total == 0, we should try to supplement with landmarks
    # The LLM + OpenAlex fallback can find field landmarks without any seed papers

    # ALWAYS require minimum landmarks for field context (even if total is high)
    refs_needed = max(0, MIN_REFS - num_refs)
    landmarks_needed = max(0, MIN_LANDMARKS - num_landmarks)

    # If we have enough of each type but total is low, add more refs (not landmarks)
    # since landmarks are the historical context and refs are for methodology
    if refs_needed == 0 and landmarks_needed == 0 and total < MIN_TOTAL:
        refs_needed = MIN_TOTAL - total

    reason = "sufficient"
    if refs_needed > 0 or landmarks_needed > 0:
        parts = []
        if refs_needed > 0:
            parts.append(f"need {refs_needed} more refs")
        if landmarks_needed > 0:
            parts.append(f"need {landmarks_needed} more landmarks")
        reason = ", ".join(parts)

    return refs_needed, landmarks_needed, reason


def suggest_papers_llm(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    existing_papers: List[Dict[str, Any]],
    num_needed: int,
    paper_type: str,  # "reference" or "landmark" or "pioneering_context"
    field: Optional[str] = None,
    is_pioneering: bool = False,
) -> List[Dict[str, Any]]:
    """
    Use LLM to suggest papers of a specific type.

    Args:
        paper_type: "reference" for methodologically similar, "landmark" for seminal papers
        is_pioneering: If True and paper_type is "landmark", use pioneering context prompt instead
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found, skipping paper suggestion")
        return []

    if num_needed <= 0:
        return []

    # Format existing papers
    existing_str = "\n".join([
        f"- {p.get('title', 'Untitled')} ({p.get('year', '?')})"
        for p in existing_papers[:10]
    ]) if existing_papers else "(none)"

    # Request MORE papers than needed to compensate for lookup failures
    # Request 2x what we need, capped at 6
    request_count = min(num_needed * 2, 6)

    # Select prompt based on paper type and pioneering status
    if paper_type == "landmark" and is_pioneering:
        # Use special prompt for pioneering works
        prompt = SUGGEST_PIONEERING_CONTEXT_PROMPT.format(
            title=title,
            abstract=(abstract or "No abstract available.")[:1500],
            year=year or 2024,
            field=field or "this research area",
            existing_papers=existing_str,
            num_needed=request_count,
        )
    elif paper_type == "landmark":
        prompt = SUGGEST_LANDMARKS_PROMPT.format(
            title=title,
            abstract=(abstract or "No abstract available.")[:1500],
            year=year or 2024,
            field=field or "this research area",
            existing_papers=existing_str,
            num_needed=request_count,
        )
    else:  # reference
        prompt = SUGGEST_REFS_PROMPT.format(
            title=title,
            abstract=(abstract or "No abstract available.")[:1500],
            year=year or 2024,
            existing_papers=existing_str,
            num_needed=request_count,
        )

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_VERSION,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic researcher. Return only valid JSON arrays.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=30.0,
                temperature=0,
            )

            content = (resp.choices[0].message.content or "").strip()

            # Use robust JSON extraction with repair (handles code blocks, extra text, malformed JSON)
            suggestions, error = extract_json_from_llm_response_with_repair(content, expected_type="array")

            if suggestions is None:
                logger.warning(f"Failed to parse LLM {paper_type} suggestions: {error}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (2 ** attempt))
                    continue
                return []

            valid = []
            for s in suggestions:
                if isinstance(s, dict) and s.get("title"):
                    valid.append({
                        "title": s["title"],
                        "authors": s.get("authors", ""),
                        "year": s.get("year"),
                        "why_relevant": s.get("why_relevant", ""),
                    })

            logger.info(f"LLM suggested {len(valid)} {paper_type} papers")
            return valid[:request_count]  # Return all suggestions - _resolve_suggestions will cap to num_needed

        except Exception as e:
            logger.warning(f"LLM {paper_type} suggestion error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
            else:
                return []

    return []


def get_subfield_from_topic_id(topic_id: Optional[str]) -> Optional[str]:
    """
    Look up the SUBFIELD name for an OpenAlex topic ID.

    OpenAlex topics have a hierarchy: Domain → Field → Subfield → Topic
    This returns the Subfield level (e.g., "Artificial Intelligence", "Computer Vision").

    Using subfield instead of field provides more granular matching:
    - Field "Computer Science" includes both NLP and CV papers
    - Subfield "Artificial Intelligence" vs "Computer Vision" distinguishes them

    Args:
        topic_id: OpenAlex topic ID (e.g., "T10123" or full URL)

    Returns:
        Subfield display name, or None if not found
    """
    if not topic_id:
        return None

    # Handle full URLs
    if "/" in topic_id:
        topic_id = topic_id.rsplit("/", 1)[-1]

    try:
        url = f"https://api.openalex.org/topics/{topic_id}"
        resp = requests.get(url, timeout=OPENALEX_TIMEOUT)
        if resp.status_code != 200:
            logger.debug(f"Topic lookup failed for {topic_id}: {resp.status_code}")
            return None

        data = resp.json()
        # Use subfield for more granular matching
        subfield_info = data.get("subfield", {})
        return subfield_info.get("display_name")
    except Exception as e:
        logger.warning(f"Topic lookup error for {topic_id}: {e}")
        return None


# Alias for backwards compatibility
def get_field_from_topic_id(topic_id: Optional[str]) -> Optional[str]:
    """Alias for get_subfield_from_topic_id for backwards compatibility."""
    return get_subfield_from_topic_id(topic_id)


def lookup_paper_in_openalex(
    title: str,
    year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Look up a paper in OpenAlex by title.

    Returns dict with: work_id, title, year, cited_by_count, abstract
    """
    if not title or len(title) < 10:
        return None

    try:
        search_title = title.replace('"', '').strip()

        url = "https://api.openalex.org/works"
        params = {
            "search": search_title,
            "per-page": 5,
        }
        if year:
            # Ensure year is int for arithmetic
            year_int = int(year) if isinstance(year, str) else year
            params["filter"] = f"publication_year:{year_int-1}-{year_int+1}"

        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])

        if not results:
            return None

        # Find best match by title similarity
        search_lower = search_title.lower()
        best_match = None
        best_score = 0

        for r in results:
            result_title = (r.get("title") or "").lower()
            words1 = set(search_lower.split())
            words2 = set(result_title.split())
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > best_score:
                    best_score = overlap
                    best_match = r

        # Require at least 50% word overlap AND the paper must have significant citations
        # to avoid false positive matches on low-quality results
        if not best_match or best_score < 0.50:
            logger.debug(f"No good match for '{title[:50]}...' (best: {best_score:.2f})")
            return None

        # For landmark papers, verify it has substantial citations
        cited_count = best_match.get("cited_by_count", 0)
        if cited_count < 100 and best_score < 0.7:
            # Low citation + low match = likely wrong paper
            logger.debug(f"Rejecting low-confidence match: '{title[:50]}...' ({cited_count} cites, {best_score:.2f} match)")
            return None

        work_id = best_match.get("id", "")
        if work_id and "/" in work_id:
            work_id = work_id.rsplit("/", 1)[-1]

        # Extract domain info for cross-domain validation
        # Use TOPIC ID for precise matching (not subfield - too coarse)
        # Subfields like "Artificial Intelligence" include both NLP and gradient boosting
        primary_topic = best_match.get("primary_topic", {})
        subfield_info = primary_topic.get("subfield", {})
        field_info = primary_topic.get("field", {})
        domain_info = primary_topic.get("domain", {})

        # Extract topic_id for precise cross-domain filtering
        topic_id = None
        topic_url = primary_topic.get("id")
        if topic_url and "/" in topic_url:
            topic_id = topic_url.rsplit("/", 1)[-1]

        return {
            "work_id": work_id,
            "title": best_match.get("title"),
            "year": best_match.get("publication_year"),
            "cited_by_count": best_match.get("cited_by_count", 0),
            "abstract": _extract_abstract(best_match),
            "field_name": subfield_info.get("display_name"),  # Keep for backwards compat
            "topic_id": topic_id,  # NEW: Topic ID for precise matching
            "domain_name": domain_info.get("display_name"),
            "broad_field": field_info.get("display_name"),  # Keep broad field for reference
        }

    except Exception as e:
        logger.warning(f"OpenAlex lookup error for '{title[:50]}...': {e}")
        return None


def _extract_abstract(work: Dict[str, Any]) -> Optional[str]:
    """Extract abstract from OpenAlex work data."""
    return decode_openalex_abstract(work.get("abstract_inverted_index"))


def _resolve_suggestions(
    suggestions: List[Dict[str, Any]],
    existing_titles: set,
    relationship: str,
    max_papers: int,
    target_field: Optional[str] = None,
    target_title: Optional[str] = None,
    target_abstract: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Look up suggestions in OpenAlex and return resolved papers (parallel).

    Args:
        target_field: OpenAlex subfield name of target paper for cross-domain filtering.
        target_title: Title of target paper for methodology filtering.
        target_abstract: Abstract of target paper for methodology filtering.
    """
    def check_methodology_mismatch(paper: Dict[str, Any]) -> bool:
        if is_methodology_mismatch(target_title, target_abstract, paper.get('title'), paper.get('abstract')):
            logger.info(f"Methodology mismatch: {paper.get('title', '')[:40]}...")
            return True
        return False

    # Filter out already-existing titles first
    to_lookup = [
        s for s in suggestions
        if s["title"].lower() not in existing_titles
    ][:max_papers + 4]  # Fetch extra in case some fail or are cross-domain

    if not to_lookup:
        return []

    resolved = []

    # Parallel OpenAlex lookups
    def lookup_one(suggestion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        paper = lookup_paper_in_openalex(
            title=suggestion["title"],
            year=suggestion.get("year"),
        )
        if paper:
            paper["relationship"] = relationship
            paper["why_relevant"] = suggestion.get("why_relevant", "")
        return paper

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(lookup_one, s): s for s in to_lookup}

        for future in as_completed(futures):
            if len(resolved) >= max_papers:
                break

            paper = future.result()
            if paper and paper["title"].lower() not in existing_titles:
                # FIRST: Check methodology mismatch (most precise filter)
                if check_methodology_mismatch(paper):
                    continue

                # SECOND: Use SUBFIELD comparison as fallback
                # Topic IDs are too specific: related papers have different topics
                paper_field = paper.get("field_name")
                if target_field and paper_field and paper_field != target_field:
                    logger.info(
                        f"Skipping cross-domain paper: {paper['title'][:40]}... "
                        f"(field: {paper_field}, target: {target_field})"
                    )
                    continue

                resolved.append(paper)
                existing_titles.add(paper["title"].lower())
                logger.info(
                    f"Resolved {relationship}: {paper['title'][:50]}... "
                    f"({paper['work_id']}, {paper['cited_by_count']} cites)"
                )

    return resolved[:max_papers]


def fetch_top_cited_papers_openalex(
    search_terms: str,
    before_year: int,
    existing_titles: set,
    limit: int = 5,
    primary_topic_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fallback: Fetch highly-cited papers directly from OpenAlex by search terms.

    This is used when LLM suggestions can't be resolved - we query OpenAlex
    directly for seminal papers in the field.

    Args:
        primary_topic_id: OpenAlex topic ID to constrain results to same domain
    """
    if not search_terms or len(search_terms) < 5:
        return []

    def _query_openalex(filters: List[str]) -> List[Dict[str, Any]]:
        """Execute OpenAlex query with given filters."""
        url = "https://api.openalex.org/works"
        params = {
            "search": search_terms[:100],
            "filter": ",".join(filters),
            "sort": "cited_by_count:desc",
            "per-page": limit * 2,
        }
        resp = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("results", [])

    try:
        # Build filter with topic constraint to prevent cross-domain results
        base_filters = [f"publication_year:<{before_year}", "cited_by_count:>300"]
        results = []

        # Try with topic filter first if available
        if primary_topic_id:
            filters_with_topic = base_filters + [f"primary_topic.id:{primary_topic_id}"]
            results = _query_openalex(filters_with_topic)

            if len(results) == 0:
                # Topic filter returned nothing - DO NOT fall back to unfiltered search
                # as it returns cross-domain papers (chemistry for statistics, etc.)
                logger.info(
                    f"Topic-filtered query returned 0 results, keeping empty to avoid cross-domain papers"
                )
                # Return empty rather than off-topic results
                results = []
            elif len(results) < limit:
                logger.info(
                    f"Topic-filtered query returned only {len(results)} results "
                    f"(keeping topic filter to avoid off-topic papers)"
                )
        else:
            # No topic filter available - skip this fallback entirely
            # Cross-domain papers are worse than no additional papers
            logger.info("No topic_id available for OpenAlex fallback, skipping to avoid cross-domain papers")
            results = []

        papers = []
        for r in results:
            if len(papers) >= limit:
                break

            paper_title = r.get("title", "")
            if not paper_title or paper_title.lower() in existing_titles:
                continue

            work_id = r.get("id", "")
            if work_id and "/" in work_id:
                work_id = work_id.rsplit("/", 1)[-1]

            papers.append({
                "work_id": work_id,
                "title": paper_title,
                "year": r.get("publication_year"),
                "cited_by_count": r.get("cited_by_count", 0),
                "abstract": _extract_abstract(r),
                "relationship": "field_landmark",
                "why_relevant": f"Highly-cited foundational paper in this field",
            })
            existing_titles.add(paper_title.lower())

        logger.info(f"OpenAlex fallback found {len(papers)} highly-cited papers")
        return papers

    except Exception as e:
        logger.warning(f"OpenAlex fallback search error: {e}")
        return []


def _map_field_to_s2_fields(
    field: Optional[str],
    title: Optional[str] = None,
    abstract: Optional[str] = None,
) -> Optional[List[str]]:
    """Map OpenAlex field/category names to Semantic Scholar fields of study.

    Returns a list of S2 field names for filtering, or None if unmapped.

    When field is generic (like "applied"), falls back to title/abstract analysis.
    """
    # Map common OpenAlex categories/fields to S2 field names
    # S2 fields: Psychology, Medicine, Biology, Computer Science, Mathematics,
    #            Economics, Physics, Chemistry, Engineering, Environmental Science, etc.
    field_mappings = {
        # Psychology/Behavioral (comprehensive - Q4 is psychology)
        "psychology": ["Psychology"],
        "cognitive science": ["Psychology", "Computer Science"],
        "behavioral": ["Psychology"],
        "psychiatry": ["Psychology", "Medicine"],
        "neuroscience": ["Psychology", "Biology", "Medicine"],
        "mental health": ["Psychology", "Medicine"],
        "depression": ["Psychology", "Medicine"],
        "anxiety": ["Psychology", "Medicine"],
        "therapy": ["Psychology", "Medicine"],
        "cognitive behavioral": ["Psychology"],

        # Medicine/Health
        "medicine": ["Medicine"],
        "medical": ["Medicine"],
        "health": ["Medicine"],
        "clinical": ["Medicine"],
        "epidemiology": ["Medicine"],
        "pharmacology": ["Medicine", "Biology"],
        "treatment": ["Medicine", "Psychology"],
        "drug": ["Medicine"],
        "therapeutic": ["Medicine", "Psychology"],
        "toxicology": ["Biology", "Medicine", "Environmental Science"],
        "toxicity": ["Biology", "Medicine", "Environmental Science"],
        "oxidative stress": ["Biology", "Medicine"],

        # Biology/Life Sciences
        "biology": ["Biology"],
        "biological": ["Biology"],
        "genetics": ["Biology"],
        "molecular": ["Biology"],
        "biochemistry": ["Biology", "Chemistry"],
        "ecology": ["Biology", "Environmental Science"],

        # Computer Science/Tech
        "computer science": ["Computer Science"],
        "machine learning": ["Computer Science"],
        "artificial intelligence": ["Computer Science"],
        "software": ["Computer Science"],

        # Statistics/Math
        "statistics": ["Mathematics"],
        "mathematics": ["Mathematics"],
        "statistical": ["Mathematics"],

        # Economics/Social
        "economics": ["Economics"],
        "economic": ["Economics"],
        "sociology": ["Sociology"],
        "political": ["Political Science"],
        "social": ["Sociology"],

        # Environmental
        "climate": ["Environmental Science"],
        "environmental": ["Environmental Science"],
        "earth": ["Environmental Science", "Geology"],
    }

    # Generic/uninformative categories that need fallback to title/abstract
    generic_categories = {"applied", "general", "other", "multidisciplinary", "mixed"}

    # Try field mapping first if field is not generic
    if field:
        field_lower = field.lower()
        if field_lower not in generic_categories:
            for key, s2_fields in field_mappings.items():
                if key in field_lower:
                    logger.debug(f"Mapped field '{field}' to S2 fields: {s2_fields}")
                    return s2_fields

    # Fallback: analyze title and abstract for domain keywords
    text_to_analyze = ""
    if title:
        text_to_analyze += title.lower() + " "
    if abstract:
        text_to_analyze += abstract.lower()[:500]  # First 500 chars

    if text_to_analyze:
        # Check environmental/climate FIRST (important for IPCC-type reports)
        # These keywords are distinctive and rarely appear in other domains
        env_keywords = {"climate", "carbon", "greenhouse", "atmospheric", "warming",
                       "emissions", "ipcc", "sea level", "ecosystem", "ice sheet"}
        if any(kw in text_to_analyze for kw in env_keywords):
            logger.debug(f"Inferred Environmental Science from title/abstract keywords")
            return ["Environmental Science"]

        # Check for biology keywords (specific molecular biology terms)
        bio_keywords = {"gene", "protein", "cell", "molecular", "dna", "rna",
                       "crispr", "genome", "expression", "mutation"}
        if any(kw in text_to_analyze for kw in bio_keywords):
            logger.debug(f"Inferred Biology from title/abstract keywords")
            return ["Biology"]

        # Check for psychology/mental health keywords
        psych_keywords = {"depression", "anxiety", "therapy", "cognitive behavioral",
                         "psychiatric", "mental health", "psychotherapy", "adolescent depression",
                         "fluoxetine", "antidepressant", "ssri", "mood disorder"}
        if any(kw in text_to_analyze for kw in psych_keywords):
            logger.debug(f"Inferred Psychology from title/abstract keywords")
            return ["Psychology", "Medicine"]

        # Check for medicine/clinical keywords
        medical_keywords = {"clinical trial", "patient", "drug", "efficacy",
                           "randomized", "placebo", "therapeutic", "treatment efficacy"}
        if any(kw in text_to_analyze for kw in medical_keywords):
            logger.debug(f"Inferred Medicine from title/abstract keywords")
            return ["Medicine"]

        # Check for CS keywords
        cs_keywords = {"algorithm", "neural network", "machine learning", "deep learning",
                      "computing", "software", "artificial intelligence"}
        if any(kw in text_to_analyze for kw in cs_keywords):
            logger.debug(f"Inferred Computer Science from title/abstract keywords")
            return ["Computer Science"]

        # Check for economics keywords
        econ_keywords = {"economic", "market", "financial", "investment", "saving",
                        "consumer", "behavioral economics", "fiscal"}
        if any(kw in text_to_analyze for kw in econ_keywords):
            logger.debug(f"Inferred Economics from title/abstract keywords")
            return ["Economics"]

        # Check for sociology/social science keywords
        sociology_keywords = {"society", "social structure", "structuration", "agency",
                             "social theory", "sociology", "sociological", "social capital",
                             "social construction", "ethnomethodology", "habitus"}
        if any(kw in text_to_analyze for kw in sociology_keywords):
            logger.debug(f"Inferred Sociology from title/abstract keywords")
            return ["Sociology"]

        # Check for ecology/environmental keywords (broader than climate-only)
        ecology_keywords = {"pollinator", "pollinators", "pollination", "biodiversity",
                           "ecosystem", "ecological", "ecology", "species", "habitat",
                           "conservation", "wildlife", "flora", "fauna", "crop"}
        if any(kw in text_to_analyze for kw in ecology_keywords):
            logger.debug(f"Inferred Biology/Environmental Science from title/abstract keywords")
            return ["Biology", "Environmental Science"]

        # Check for primate/animal cognition keywords
        cognition_keywords = {"chimpanzee", "primate", "ape", "cognition", "theory of mind",
                             "mental state", "attribution", "animal behavior", "ethology"}
        if any(kw in text_to_analyze for kw in cognition_keywords):
            logger.debug(f"Inferred Psychology from title/abstract keywords (animal cognition)")
            return ["Psychology"]

        # Check for toxicology keywords (important for papers like Metals/Oxidative Stress)
        toxicology_keywords = {"toxicity", "toxic", "oxidative stress", "metals", "heavy metal",
                              "cadmium", "lead", "mercury", "arsenic", "oxidant", "antioxidant"}
        if any(kw in text_to_analyze for kw in toxicology_keywords):
            logger.debug(f"Inferred Biology/Medicine from title/abstract keywords (toxicology)")
            return ["Biology", "Medicine"]

    logger.debug(f"Could not map field '{field}' to S2 fields")
    return None


def _extract_search_terms(title: str, abstract: Optional[str], field: Optional[str] = None) -> str:
    """Extract key search terms from title, abstract, and field.

    Improved extraction that:
    1. Filters out report/administrative/filler words
    2. Prioritizes domain-specific terminology
    3. Keeps compound terms together when possible
    """
    import re

    # Comprehensive stopwords including filler/generic words
    stopwords = {
        # Common words
        'this', 'that', 'with', 'from', 'have', 'been', 'were', 'which',
        'their', 'there', 'these', 'those', 'about', 'into', 'more', 'some',
        'such', 'than', 'they', 'will', 'would', 'could', 'should', 'also',
        'paper', 'method', 'approach', 'results', 'using', 'based', 'novel',
        # Report/administrative terms
        'report', 'contribution', 'working', 'group', 'assessment', 'basis',
        'first', 'second', 'third', 'fourth', 'fifth', 'part', 'volume',
        'chapter', 'section', 'panel', 'committee', 'intergovernmental',
        'national', 'international', 'institute', 'program', 'programme',
        # Filler/generic terms that dilute search (critical for Q4 type papers)
        'offered', 'most', 'favorable', 'tradeoff', 'benefit', 'risk',
        'combination', 'study', 'analysis', 'research', 'findings',
        'showed', 'found', 'suggest', 'suggests', 'indicate', 'indicates',
        'among', 'between', 'during', 'after', 'before', 'within',
        'both', 'only', 'many', 'each', 'other', 'various', 'different',
        'important', 'significant', 'effective', 'improved', 'better',
    }

    text = title
    if abstract:
        # Get first sentence of abstract (usually contains key concepts)
        first_sentence = abstract.split('.')[0] if '.' in abstract else abstract[:200]
        text += " " + first_sentence
    elif field:
        # If no abstract, add field context for better search
        text += " " + field

    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stopwords]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    # Return top unique words - prioritize terms from title (more important)
    return " ".join(unique[:8])  # Reduced to 8 to focus on core terms


# ============================================================================
# Multi-Source Fallback Functions (Semantic Scholar + ArXiv)
# ============================================================================

def search_semantic_scholar_by_keywords(
    search_terms: str,
    before_year: int,
    existing_titles: set,
    limit: int = 5,
    fields_of_study: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar using bulk endpoint with retry logic.

    Uses the same bulk endpoint as Feature 1/2 retrieval for better results
    and proper pagination. Falls back to regular endpoint on 400 errors.

    Args:
        search_terms: Keywords to search for
        before_year: Only return papers published before this year
        existing_titles: Titles to exclude (already have)
        limit: Maximum number of papers to return
        fields_of_study: Optional list of S2 fields to filter (e.g., ["Psychology", "Medicine"])

    Returns:
        List of paper dicts with: work_id, title, year, cited_by_count, abstract, relationship
    """
    if not search_terms or len(search_terms) < 5:
        return []

    headers = {}
    s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if s2_api_key:
        headers["x-api-key"] = s2_api_key

    # Use bulk endpoint (same as F1/F2) — better pagination and results
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {
        "query": search_terms[:200],
        "fields": "title,abstract,year,citationCount,externalIds,s2FieldsOfStudy",
        "limit": min(1000, limit * 3),
    }
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    for attempt in range(MAX_RETRIES + 1):
        try:
            S2_RATE_LIMITER.wait()
            resp = requests.get(url, params=params, headers=headers, timeout=EXTERNAL_API_TIMEOUT)

            # 400 = query too broad for bulk, fall back to regular endpoint
            if resp.status_code == 400:
                logger.info("S2 bulk returned 400, falling back to regular search")
                return _search_s2_regular(
                    search_terms, before_year, existing_titles, limit, fields_of_study, headers
                )

            # 429 = rate limited, retry with exponential backoff (like F1/F2)
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    backoff = 2 ** (attempt + 1)
                    logger.info(f"S2 bulk rate limited (429), retry {attempt+1}/{MAX_RETRIES} after {backoff}s")
                    time.sleep(backoff)
                    continue
                logger.warning("S2 bulk rate limited, all retries exhausted")
                return []

            resp.raise_for_status()
            data = resp.json()

            return _filter_s2_results(
                data.get("data", []), before_year, existing_titles, limit, fields_of_study
            )

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"S2 bulk search error: {e}, retrying in {backoff}s")
                time.sleep(backoff)
                continue
            logger.warning(f"S2 bulk search failed after retries: {e}")
            return []
        except Exception as e:
            logger.warning(f"S2 bulk search unexpected error: {e}")
            return []

    return []


def _search_s2_regular(
    search_terms: str,
    before_year: int,
    existing_titles: set,
    limit: int = 5,
    fields_of_study: Optional[List[str]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Fallback to regular S2 search endpoint when bulk returns 400."""
    try:
        S2_RATE_LIMITER.wait()
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": search_terms[:100],
            "limit": min(100, limit * 3),
            "fields": "title,abstract,year,citationCount,externalIds,s2FieldsOfStudy",
            "year": f"-{before_year - 1}",
        }
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        resp = requests.get(url, params=params, headers=headers or {}, timeout=EXTERNAL_API_TIMEOUT)
        if resp.status_code == 429:
            logger.warning("S2 regular search rate limited (429)")
            return []
        resp.raise_for_status()

        data = resp.json()
        return _filter_s2_results(
            data.get("data", []), before_year, existing_titles, limit, fields_of_study
        )
    except Exception as e:
        logger.warning(f"S2 regular search error: {e}")
        return []


def _filter_s2_results(
    raw_papers: List[Dict[str, Any]],
    before_year: int,
    existing_titles: set,
    limit: int,
    fields_of_study: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter and format S2 API results into grounding paper format."""
    papers = []

    for paper in raw_papers:
        if len(papers) >= limit:
            break

        paper_title = paper.get("title", "")
        if not paper_title or paper_title.lower() in existing_titles:
            continue

        citation_count = paper.get("citationCount", 0)
        if citation_count < 30:
            continue

        paper_year = paper.get("year")
        if paper_year and paper_year >= before_year:
            continue

        # Validate paper's actual field matches requested fields
        if fields_of_study:
            paper_fields = paper.get("s2FieldsOfStudy", [])
            paper_field_names = {f.get("category", "").lower() for f in paper_fields if f.get("category")}
            requested_lower = {f.lower() for f in fields_of_study}
            if not paper_field_names.intersection(requested_lower):
                logger.debug(
                    f"Skipping S2 paper '{paper_title[:40]}...' - "
                    f"fields {paper_field_names} don't match {requested_lower}"
                )
                continue

        # Extract OpenAlex ID from externalIds (like F1 pattern)
        external_ids = paper.get("externalIds", {}) or {}
        openalex_id = external_ids.get("OpenAlex")
        doi = external_ids.get("DOI")
        paper_id = paper.get("paperId", "")

        work_id = openalex_id if openalex_id else f"S2:{paper_id}"

        papers.append({
            "work_id": work_id,
            "title": paper_title,
            "year": paper_year,
            "cited_by_count": citation_count,
            "abstract": paper.get("abstract"),
            "relationship": "field_landmark",
            "why_relevant": "Highly-cited foundational paper (via Semantic Scholar)",
            "doi": doi,
        })
        existing_titles.add(paper_title.lower())

    logger.info(f"Semantic Scholar search found {len(papers)} highly-cited papers")
    return papers


def _map_s2_fields_to_arxiv_category(fields_of_study: Optional[List[str]]) -> Optional[str]:
    """Map S2 field names to ArXiv category prefixes for filtering."""
    if not fields_of_study:
        return None
    mapping = {
        "computer science": "cs",
        "mathematics": "math",
        "physics": "physics",
        "statistics": "stat",
        "quantitative biology": "q-bio",
        "quantitative finance": "q-fin",
        "electrical engineering": "eess",
    }
    for field in fields_of_study:
        cat = mapping.get(field.lower())
        if cat:
            return cat
    return None


def search_arxiv_by_keywords(
    search_terms: str,
    before_year: int,
    existing_titles: set,
    limit: int = 5,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search ArXiv for papers by keywords with 429 retry handling.

    Best for: CS, Physics, Math, Statistics papers.

    Args:
        search_terms: Keywords to search for
        before_year: Only return papers published before this year
        existing_titles: Titles to exclude
        limit: Maximum papers to return
        category: Optional ArXiv category (e.g., "cs.LG", "cs", "stat.ML")
    """
    if not search_terms or len(search_terms) < 5:
        return []

    import re

    # Build search query with category filter
    search_query = f"all:{search_terms}"
    if category:
        search_query = f"cat:{category} AND {search_query}"

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": search_query,
        "max_results": min(50, limit * 3),  # Cap at 50 to avoid ArXiv timeouts (F2 lesson)
        "sortBy": "relevance",
    }

    # Retry loop with 429 handling (like F2 pattern)
    for attempt in range(MAX_RETRIES + 1):
        try:
            ARXIV_RATE_LIMITER.wait()
            resp = requests.get(url, params=params, timeout=EXTERNAL_API_TIMEOUT)

            # 429: rate limited — wait and retry (ArXiv frequently returns these)
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    retry_after = int(resp.headers.get("Retry-After", 15))
                    retry_after = max(retry_after, 10)  # At least 10s for ArXiv
                    logger.warning(f"ArXiv 429, waiting {retry_after}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(retry_after)
                    continue
                logger.warning("ArXiv rate limited, all retries exhausted")
                return []

            resp.raise_for_status()

            content = resp.text
            papers = []
            entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)

            for entry in entries:
                if len(papers) >= limit:
                    break

                title_match = re.search(r'<title[^>]*>(.*?)</title>', entry, re.DOTALL)
                if not title_match:
                    continue
                paper_title = title_match.group(1).strip()
                paper_title = re.sub(r'\s+', ' ', paper_title)

                if paper_title.lower() in existing_titles:
                    continue

                published_match = re.search(r'<published>(\d{4})', entry)
                year = int(published_match.group(1)) if published_match else None

                if year and year >= before_year:
                    continue

                abstract_match = re.search(r'<summary[^>]*>(.*?)</summary>', entry, re.DOTALL)
                abstract = abstract_match.group(1).strip() if abstract_match else None

                id_match = re.search(r'<id>https?://arxiv\.org/abs/([^<]+)</id>', entry)
                arxiv_id = id_match.group(1) if id_match else None

                papers.append({
                    "work_id": f"ArXiv:{arxiv_id}" if arxiv_id else "ArXiv:unknown",
                    "title": paper_title,
                    "year": year,
                    "cited_by_count": 0,  # ArXiv doesn't provide citation counts
                    "abstract": abstract,
                    "relationship": "field_landmark",
                    "why_relevant": "Related foundational paper (via ArXiv)",
                    "arxiv_id": arxiv_id,
                })
                existing_titles.add(paper_title.lower())

            logger.info(f"ArXiv search found {len(papers)} papers")
            return papers

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"ArXiv search error: {e}, retrying in {backoff}s")
                time.sleep(backoff)
                continue
            logger.warning(f"ArXiv search failed after retries: {e}")
            return []
        except Exception as e:
            logger.warning(f"ArXiv search error: {e}")
            return []

    return []


def resolve_paper_to_openalex(
    paper: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Try to resolve a paper from S2/ArXiv to OpenAlex for consistent work_id.

    Returns the paper with updated work_id and field_name if found in OpenAlex, else original.
    """
    # Try by DOI first (most reliable)
    doi = paper.get("doi")
    if doi:
        try:
            url = f"https://api.openalex.org/works/doi:{doi}"
            resp = requests.get(url, timeout=OPENALEX_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                work_id = data.get("id", "")
                if work_id and "/" in work_id:
                    work_id = work_id.rsplit("/", 1)[-1]
                paper["work_id"] = work_id
                paper["cited_by_count"] = data.get("cited_by_count", paper.get("cited_by_count", 0))
                # Extract field_name and topic_id for cross-domain filtering
                primary_topic = data.get("primary_topic", {})
                subfield_info = primary_topic.get("subfield", {})
                if subfield_info.get("display_name"):
                    paper["field_name"] = subfield_info["display_name"]
                # Extract topic_id for precise matching
                topic_url = primary_topic.get("id")
                if topic_url and "/" in topic_url:
                    paper["topic_id"] = topic_url.rsplit("/", 1)[-1]
                return paper
        except Exception:
            pass

    # Fallback: search by title
    result = lookup_paper_in_openalex(paper.get("title", ""), paper.get("year"))
    if result:
        paper["work_id"] = result["work_id"]
        paper["cited_by_count"] = result.get("cited_by_count", paper.get("cited_by_count", 0))
        paper["abstract"] = result.get("abstract") or paper.get("abstract")
        # Copy field_name and topic_id for cross-domain filtering
        if result.get("field_name"):
            paper["field_name"] = result["field_name"]
        if result.get("topic_id"):
            paper["topic_id"] = result["topic_id"]

    return paper


def fetch_papers_multi_source(
    search_terms: str,
    before_year: int,
    existing_titles: set,
    limit: int = 5,
    primary_topic_id: Optional[str] = None,
    fields_of_study: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-source parallel retrieval for finding grounding papers.

    Searches ALL CO-EQUAL sources in PARALLEL:
    - OpenAlex (with topic filter if available)
    - Semantic Scholar
    - ArXiv (for CS/physics/stats topics)

    Results are combined and deduplicated by title.

    Returns combined deduplicated results up to limit.
    """
    import threading

    # Thread-safe containers for results
    openalex_papers: List[Dict[str, Any]] = []
    s2_papers: List[Dict[str, Any]] = []
    arxiv_papers: List[Dict[str, Any]] = []
    openalex_lock = threading.Lock()
    s2_lock = threading.Lock()
    arxiv_lock = threading.Lock()

    def search_openalex():
        """Search OpenAlex."""
        try:
            # Copy existing_titles to avoid thread conflicts
            local_titles = existing_titles.copy()
            results = fetch_top_cited_papers_openalex(
                search_terms=search_terms,
                before_year=before_year,
                existing_titles=local_titles,
                limit=limit,
                primary_topic_id=primary_topic_id,
            )
            if results:
                with openalex_lock:
                    openalex_papers.extend(results)
                logger.info(f"OpenAlex search: {len(results)} papers found")
        except Exception as e:
            logger.warning(f"OpenAlex search error: {e}")

    def search_s2():
        """Search Semantic Scholar."""
        try:
            local_titles = existing_titles.copy()
            results = search_semantic_scholar_by_keywords(
                search_terms=search_terms,
                before_year=before_year,
                existing_titles=local_titles,
                limit=limit,
                fields_of_study=fields_of_study,
            )
            if results:
                with s2_lock:
                    s2_papers.extend(results)
                logger.info(f"Semantic Scholar search: {len(results)} papers found")
        except Exception as e:
            logger.warning(f"Semantic Scholar search error: {e}")

    def search_arxiv():
        """Search ArXiv (for technical topics) with category filter."""
        try:
            local_titles = existing_titles.copy()
            # Derive ArXiv category from S2 fields (e.g., Computer Science → "cs")
            arxiv_cat = _map_s2_fields_to_arxiv_category(fields_of_study)
            results = search_arxiv_by_keywords(
                search_terms=search_terms,
                before_year=before_year,
                existing_titles=local_titles,
                limit=limit,
                category=arxiv_cat,
            )
            if results:
                with arxiv_lock:
                    arxiv_papers.extend(results)
                logger.info(f"ArXiv search: {len(results)} papers found")
        except Exception as e:
            logger.warning(f"ArXiv search error: {e}")

    # Execute all searches in PARALLEL (like feature2 retrieval)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(search_openalex),
            executor.submit(search_s2),
            executor.submit(search_arxiv),
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Search failed: {e}")

    logger.info(
        f"Multi-source parallel results: OpenAlex={len(openalex_papers)}, "
        f"S2={len(s2_papers)}, ArXiv={len(arxiv_papers)}"
    )

    # Combine and deduplicate results
    # Priority: OpenAlex > Semantic Scholar > ArXiv (OpenAlex has best work_ids)
    combined_papers: List[Dict[str, Any]] = []
    seen_titles: set = existing_titles.copy()

    # Add OpenAlex papers first (already have proper work_ids)
    for p in openalex_papers:
        title_lower = p.get("title", "").lower()
        if title_lower and title_lower not in seen_titles:
            combined_papers.append(p)
            seen_titles.add(title_lower)

    # Add Semantic Scholar papers, resolving to OpenAlex IDs when possible
    for p in s2_papers:
        title_lower = p.get("title", "").lower()
        if title_lower and title_lower not in seen_titles:
            # Try to resolve to OpenAlex work_id
            if p.get("work_id", "").startswith("S2:"):
                resolved = resolve_paper_to_openalex(p)
                combined_papers.append(resolved if resolved else p)
            else:
                combined_papers.append(p)
            seen_titles.add(title_lower)

    # Add ArXiv papers, resolving to OpenAlex IDs when possible
    for p in arxiv_papers:
        title_lower = p.get("title", "").lower()
        if title_lower and title_lower not in seen_titles:
            resolved = resolve_paper_to_openalex(p)
            combined_papers.append(resolved if resolved else p)
            seen_titles.add(title_lower)

    # Sort by citation count (prefer more cited papers)
    combined_papers.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)

    logger.info(f"Multi-source combined: {len(combined_papers)} unique papers (returning up to {limit})")
    return combined_papers[:limit]


# ============================================================================
# Main Entry Point
# ============================================================================

def supplement_grounding_papers(
    title: str,
    abstract: Optional[str],
    year: Optional[int],
    existing_refs: List[Dict[str, Any]],
    existing_landmarks: List[Dict[str, Any]],
    field: Optional[str] = None,
    primary_topic_id: Optional[str] = None,
    is_pioneering: bool = False,
    target_work_id: Optional[str] = None,
    conn: Optional[Connection] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Supplement grounding papers to ensure balanced coverage.

    Args:
        title: Target paper title
        abstract: Target paper abstract
        year: Target paper year
        existing_refs: Referenced works already found
        existing_landmarks: Landmark papers already found
        field: Optional field/topic description for better landmark suggestions
        primary_topic_id: OpenAlex topic ID to constrain fallback search to same domain
        is_pioneering: If True, use special pioneering logic (1 before + 3 citing)
        target_work_id: OpenAlex work ID of target paper (needed for citing papers)
        conn: Optional database connection for landmark caching

    Returns:
        (additional_refs, additional_landmarks) - may be empty lists
    """
    refs_needed, landmarks_needed, reason = assess_grounding_needs(
        existing_refs, existing_landmarks
    )

    # Check landmark cache if connection provided and landmarks are needed
    use_cached_landmarks = False
    cached_landmark_list: List[Dict[str, Any]] = []
    if conn is not None and target_work_id and landmarks_needed > 0:
        cached_landmarks = get_landmarks(conn, target_work_id)
        if cached_landmarks:
            # Filter out any landmarks already in existing_landmarks
            existing_titles_for_filter = {p.get("title", "").lower() for p in existing_landmarks}
            existing_titles_for_filter.add(title.lower())  # Exclude target paper
            filtered_cached = [
                lm for lm in cached_landmarks
                if lm.get("title", "").lower() not in existing_titles_for_filter
            ]
            if filtered_cached:
                logger.info(f"Using {len(filtered_cached)} cached landmarks for {target_work_id}")
                use_cached_landmarks = True
                cached_landmark_list = filtered_cached[:landmarks_needed]

    if reason == "no_grounding_unavailable":
        logger.debug("No existing grounding papers - UNAVAILABLE case, not supplementing")
        return [], []

    if reason == "sufficient":
        logger.debug(f"Grounding sufficient: {len(existing_refs)} refs, {len(existing_landmarks)} landmarks")
        return [], []

    logger.info(f"Supplementing grounding: {reason}")

    # GENERAL FIX: Get the target paper's OpenAlex SUBFIELD name from its topic_id
    # Using subfield instead of field provides more granular matching:
    # - Field "Computer Science" is too broad (includes both NLP and CV)
    # - Subfield "Artificial Intelligence" vs "Computer Vision" distinguishes them
    target_field_name: Optional[str] = None
    if primary_topic_id:
        target_field_name = get_subfield_from_topic_id(primary_topic_id)
        if target_field_name:
            logger.info(f"Target paper subfield: {target_field_name} (from topic {primary_topic_id})")
        else:
            logger.debug(f"Could not resolve subfield for topic {primary_topic_id}")

    # Track existing titles AND work_ids to avoid duplicates
    # IMPORTANT: Include the target paper's title to prevent self-citation
    existing_titles = {
        p.get("title", "").lower()
        for p in existing_refs + existing_landmarks
    }
    existing_titles.add(title.lower())  # Exclude target paper from suggestions

    # Also track work_ids for more robust deduplication
    existing_work_ids = {
        p.get("work_id")
        for p in existing_refs + existing_landmarks
        if p.get("work_id")
    }
    if target_work_id:
        existing_work_ids.add(target_work_id)

    additional_refs = []
    additional_landmarks = []

    # Suggest reference-type papers if needed
    if refs_needed > 0:
        logger.info(f"Requesting {refs_needed} reference-type papers from LLM")
        suggestions = suggest_papers_llm(
            title=title,
            abstract=abstract,
            year=year,
            existing_papers=existing_refs + existing_landmarks,
            num_needed=refs_needed,
            paper_type="reference",
        )
        additional_refs = _resolve_suggestions(
            suggestions, existing_titles, "cited_reference", refs_needed,
            target_field=target_field_name,
            target_title=title,
            target_abstract=abstract,
        )

    # SPECIAL HANDLING FOR PIONEERING WORKS
    # For pioneering papers: focus on papers from BEFORE to show historical context
    # All grounding papers should be from BEFORE the target year for proper novelty assessment
    if is_pioneering and target_work_id and landmarks_needed > 0:
        if use_cached_landmarks:
            # Use cached landmarks instead of LLM generation
            additional_landmarks = cached_landmark_list
            logger.info(f"Using {len(additional_landmarks)} cached landmarks for pioneering work")
        else:
            logger.info(f"Pioneering work detected - requesting {landmarks_needed} predecessor papers")

            # Get papers from BEFORE the pioneering work (historical context)
            suggestions = suggest_papers_llm(
                title=title,
                abstract=abstract,
                year=year,
                existing_papers=existing_refs + existing_landmarks + additional_refs,
                num_needed=landmarks_needed,
                paper_type="landmark",
                field=field,
                is_pioneering=True,
            )
            predecessor_papers = _resolve_suggestions(
                suggestions, existing_titles, "field_landmark", landmarks_needed,
                target_field=target_field_name,
                target_title=title,
                target_abstract=abstract,
            )
            if predecessor_papers:
                for p in predecessor_papers:
                    p["why_relevant"] = f"Predecessor work showing state of field before this pioneering paper"
                additional_landmarks.extend(predecessor_papers)
                logger.info(f"Found {len(predecessor_papers)} predecessor paper(s)")

            # If we still don't have enough landmarks, use multi-source parallel search
            # with papers from BEFORE the target year
            total_landmarks = len(existing_landmarks) + len(additional_landmarks)
            if total_landmarks < MIN_LANDMARKS:
                still_needed = MIN_LANDMARKS - total_landmarks
                logger.info(f"Still need {still_needed} more landmarks - trying multi-source search")
                search_terms = _extract_search_terms(title, abstract, field)
                s2_fields = _map_field_to_s2_fields(field, title=title, abstract=abstract)
                # Request extra as buffer for deduplication/filtering
                request_limit = max(still_needed * 2, 4)
                fallback_papers = fetch_papers_multi_source(
                    search_terms=search_terms,
                    before_year=year or 2024,  # Papers from BEFORE target year for context
                    existing_titles=existing_titles,
                    limit=request_limit,
                    primary_topic_id=primary_topic_id,
                    fields_of_study=s2_fields,
                )
                # Take all - final deduplication will filter and cap at MAX_TOTAL
                additional_landmarks.extend(fallback_papers)

    # STANDARD HANDLING FOR NON-PIONEERING WORKS
    elif landmarks_needed > 0:
        if use_cached_landmarks:
            # Use cached landmarks instead of LLM generation
            additional_landmarks = cached_landmark_list
            logger.info(f"Using {len(additional_landmarks)} cached landmarks")
        else:
            logger.info(f"Requesting {landmarks_needed} landmark-type papers from LLM")
            suggestions = suggest_papers_llm(
                title=title,
                abstract=abstract,
                year=year,
                existing_papers=existing_refs + existing_landmarks + additional_refs,
                num_needed=landmarks_needed,
                paper_type="landmark",
                field=field,
                is_pioneering=False,
            )
            additional_landmarks = _resolve_suggestions(
                suggestions, existing_titles, "field_landmark", landmarks_needed,
                target_field=target_field_name,
                target_title=title,
                target_abstract=abstract,
            )

            # FALLBACK: If LLM suggestions didn't resolve enough landmarks,
            # search all sources in parallel for highly-cited papers in the field
            still_needed = landmarks_needed - len(additional_landmarks)
            if still_needed > 0:
                logger.info(f"LLM resolved {len(additional_landmarks)} landmarks, need {still_needed} more - trying multi-source search")
                search_terms = _extract_search_terms(title, abstract, field)
                s2_fields = _map_field_to_s2_fields(field, title=title, abstract=abstract)
                # Request extra as buffer for deduplication/filtering
                # Request 2x what we need to account for duplicates being filtered later
                request_limit = max(still_needed * 2, 4)
                fallback_papers = fetch_papers_multi_source(
                    search_terms=search_terms,
                    before_year=year or 2024,
                    existing_titles=existing_titles,
                    limit=request_limit,
                    primary_topic_id=primary_topic_id,
                    fields_of_study=s2_fields,
                )
                # Take all - final deduplication will filter and cap at MAX_TOTAL
                additional_landmarks.extend(fallback_papers)

    # FINAL GUARANTEE: Ensure we have at least MIN_TOTAL grounding papers
    total_existing = len(existing_refs) + len(existing_landmarks)
    total_additional = len(additional_refs) + len(additional_landmarks)
    total_all = total_existing + total_additional

    if total_all < MIN_TOTAL:
        still_missing = MIN_TOTAL - total_all
        logger.info(f"Need {still_missing} more papers to reach {MIN_TOTAL} - trying domain-specific S2 search")

        # LAST RESORT: Domain-specific Semantic Scholar search with multiple query strategies
        # This is safer than unfiltered OpenAlex because S2 fields are domain-aware
        s2_fields = _map_field_to_s2_fields(field, title=title, abstract=abstract)
        if s2_fields:
            # Try multiple search strategies
            search_queries = [
                " ".join(title.split()[:5]),  # First 5 words
                " ".join(title.split()[:8]),  # First 8 words (broader)
                _extract_search_terms(title, None, field),  # Keyword extraction
            ]
            # Add field name to queries if available
            if s2_fields and len(s2_fields) > 0:
                search_queries.append(f"{s2_fields[0]} {title.split()[0]}")  # Field + first word

            for query in search_queries:
                if total_all >= MIN_TOTAL:
                    break
                try:
                    last_resort_papers = search_semantic_scholar_by_keywords(
                        search_terms=query,
                        before_year=year or 2024,
                        existing_titles=existing_titles,
                        limit=max(still_missing * 2, 4),
                        fields_of_study=s2_fields,
                    )
                    if last_resort_papers:
                        logger.info(f"Last-resort S2 search '{query[:30]}...' found {len(last_resort_papers)} papers")
                        # Update existing_titles to avoid duplicates in next iteration
                        for p in last_resort_papers:
                            existing_titles.add(p.get("title", "").lower())
                        additional_landmarks.extend(last_resort_papers)
                        # Recalculate totals
                        total_additional = len(additional_refs) + len(additional_landmarks)
                        total_all = total_existing + total_additional
                except Exception as e:
                    logger.warning(f"Last-resort S2 search failed for '{query[:30]}...': {e}")

        # Recalculate totals after last resort
        total_additional = len(additional_refs) + len(additional_landmarks)
        total_all = total_existing + total_additional

        if total_all < MIN_TOTAL:
            logger.warning(
                f"After all attempts, only have {total_all} grounding papers "
                f"(need {MIN_TOTAL}). Accepting fewer to avoid cross-domain contamination."
            )

    # Use centralized methodology mismatch detection
    def _check_methodology_mismatch(paper: Dict[str, Any]) -> bool:
        """Check if a paper uses a different methodology than the target."""
        if is_methodology_mismatch(title, abstract, paper.get('title'), paper.get('abstract')):
            logger.debug(f"Methodology mismatch: {paper.get('title', '')[:40]}...")
            return True
        return False

    # Cross-domain filtering using SUBFIELD (not topic_id - too strict)
    # Topic IDs are too specific: related papers have different topics
    # Stale DB data (wrong topics like XGBoost) is caught by landmark_retrieval verification
    def _is_cross_domain(paper: Dict[str, Any]) -> bool:
        # First check methodology mismatch (more specific than subfield)
        if _check_methodology_mismatch(paper):
            return True

        if not target_field_name:
            return False

        paper_field = paper.get("field_name")  # Set when resolved to OpenAlex
        work_id = paper.get("work_id", "")

        # If paper has field_name, compare directly
        if paper_field:
            if paper_field != target_field_name:
                logger.debug(f"Cross-domain (subfield): {paper_field} != {target_field_name}")
                return True
            return False

        # Paper has no field_name - it wasn't resolved to OpenAlex
        # S2/ArXiv field filtering is too loose (keyword matching), so exclude
        # unresolved papers when we have a target field to prevent contamination
        if work_id.startswith("S2:") or work_id.startswith("ArXiv:"):
            logger.debug(f"Excluding unresolved {work_id[:20]}... (no OpenAlex field_name)")
            return True

        return False

    # Filter refs and landmarks - remove cross-domain and duplicates by work_id
    def _is_duplicate(paper: Dict[str, Any]) -> bool:
        work_id = paper.get("work_id")
        return work_id and work_id in existing_work_ids

    filtered_refs = [
        r for r in additional_refs
        if not _is_cross_domain(r) and not _is_duplicate(r)
    ]
    filtered_landmarks = [
        lm for lm in additional_landmarks
        if not _is_cross_domain(lm) and not _is_duplicate(lm)
    ]

    removed_refs = len(additional_refs) - len(filtered_refs)
    removed_landmarks = len(additional_landmarks) - len(filtered_landmarks)
    if removed_refs > 0 or removed_landmarks > 0:
        logger.info(f"Removed {removed_refs} refs (cross-domain/duplicate), {removed_landmarks} landmarks (cross-domain/duplicate)")

    # Cap total grounding papers at MAX_TOTAL to avoid over-grounding
    total_existing = len(existing_refs) + len(existing_landmarks)
    total_new = len(filtered_refs) + len(filtered_landmarks)
    total_all = total_existing + total_new

    if total_all > MAX_TOTAL:
        excess = total_all - MAX_TOTAL
        logger.info(f"Capping grounding at {MAX_TOTAL} (was {total_all})")
        # Remove excess from landmarks first (they're less specific than refs)
        if len(filtered_landmarks) > excess:
            filtered_landmarks = filtered_landmarks[:-excess]
        else:
            # Remove all excess landmarks, then some refs if needed
            remaining = excess - len(filtered_landmarks)
            filtered_landmarks = []
            if remaining > 0 and len(filtered_refs) > remaining:
                filtered_refs = filtered_refs[:-remaining]

    logger.info(
        f"Supplemented: +{len(filtered_refs)} refs, +{len(filtered_landmarks)} landmarks"
    )

    # Cache landmarks for future use (if not already from cache) - only cache filtered results
    if conn is not None and target_work_id and filtered_landmarks and not use_cached_landmarks:
        # Combine existing_landmarks with filtered_landmarks for complete cache
        all_landmarks = existing_landmarks + filtered_landmarks
        cache_landmarks(conn, target_work_id, all_landmarks, merge_with_existing=False)

    return filtered_refs, filtered_landmarks
