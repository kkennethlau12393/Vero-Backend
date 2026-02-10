"""
Domain-specific filtering for query-aware ranking.

This module provides domain vocabulary for boosting and downranking papers
based on detected query domain. The primary use case is civil engineering
queries, where papers about bridges/buildings/earthquake should rank higher
than papers about composites/robotics/aerospace.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Configuration for a specific domain filter."""
    name: str
    detection_terms: Set[str]  # Terms that indicate this domain in a query
    boost_terms: Set[str]      # Terms in papers that should boost relevance
    downrank_terms: Set[str]   # Terms in papers that should reduce relevance


# Civil Engineering domain configuration
CIVIL_ENGINEERING = DomainConfig(
    name="civil_engineering",
    detection_terms={
        # Explicit domain
        "civil engineering", "civil structures", "civil infrastructure",
        # Structural types
        "bridge", "bridges", "building", "buildings", "tower", "towers",
        "dam", "dams", "foundation", "foundations", "pile", "piles",
        # Phenomena specific to civil
        "earthquake", "seismic", "wind loading", "wind load",
        "structural dynamics", "soil-structure", "soil structure",
    },
    boost_terms={
        # Structural types
        "bridge", "building", "tower", "dam", "foundation", "pile",
        "high-rise", "skyscraper", "highway", "viaduct", "pier",
        # Phenomena
        "earthquake", "seismic", "wind", "vibration", "damping",
        "ground motion", "base isolation", "tuned mass damper",
        # Methods relevant to civil
        "structural dynamics", "modal analysis", "response spectra",
        "time integration", "numerical dissipation", "newmark",
        "time-history", "pushover", "capacity spectrum",
        # Systems
        "civil structure", "infrastructure", "soil-structure",
        "soil-pile", "pile-soil", "liquefaction",
        # Monitoring
        "structural health monitoring", "shm", "damage detection",
    },
    downrank_terms={
        # Aerospace/mechanical (unless explicitly civil)
        "composite shell", "laminated composite", "laminate",
        "aerospace", "aircraft", "wing", "fuselage", "propeller",
        # Robotics
        "robotic", "robot", "manipulator", "robot arm", "actuator control",
        "end-effector", "robotic arm", "industrial robot",
        # Nanomaterials (not typical for civil scale)
        "nanotube", "carbon nanotube", "graphene", "nanoscale",
        "nanocomposite", "nano-", "molecular dynamics",
        # Other domains
        "turbine blade", "rotor dynamics", "helicopter",
        "spacecraft", "satellite", "rocket",
        # Biomedical
        "biomedical", "implant", "prosthetic", "biomechanics",
        "bone", "tissue",
        # Chemistry/materials not relevant to civil
        "paraffin", "n-paraffin", "wax", "phase change material",
        "vibrational analysis", "molecular vibration", "infrared spectr",
        "raman spectr", "spectroscopy", "molecule", "molecular",
        "crystal structure", "crystalline", "crystallography",
        "chemical bond", "chemical reaction", "synthesis",
        # Cellular/lattice materials (non-structural)
        "lattice crushing", "honeycomb core", "foam core",
        "auxetic", "metamaterial", "cellular solid",
        "graded lattice", "lattice structure", "unit cell",
        # Polymers/composites manufacturing
        "polymer matrix", "epoxy resin", "curing process",
        "fiber layup", "autoclave", "polymer",
        # Energy systems (not structural)
        "battery", "fuel cell", "solar cell", "photovoltaic",
        "lithium", "electrolyte",
        # Electronics/MEMS
        "mems", "microelectromechanical", "piezoelectric harvester",
        "sensor network", "iot sensor",
        # Automotive (unless structural)
        "vehicle crash", "crashworthiness", "car body",
        "automobile", "automotive",
    },
)

# Registry of all domain configurations
DOMAIN_REGISTRY: Dict[str, DomainConfig] = {
    "civil_engineering": CIVIL_ENGINEERING,
}


def detect_domain(query_text: str) -> Optional[str]:
    """Detect the domain of a query based on vocabulary matching.

    Returns the domain name if detected, None otherwise.
    Currently supports: civil_engineering
    """
    if not query_text:
        return None

    query_lower = query_text.lower()

    for domain_name, config in DOMAIN_REGISTRY.items():
        for term in config.detection_terms:
            if term in query_lower:
                logger.info(f"Detected domain '{domain_name}' from term '{term}'")
                return domain_name

    return None


def compute_domain_alignment(
    domain: Optional[str],
    title: str,
    abstract: Optional[str] = None,
) -> float:
    """Compute domain alignment score for a paper (0.0 to 1.0).

    When a domain IS detected, scoring is harsh to push off-domain papers down:
        - 1.0: Clear on-domain (boost terms, no downrank)
        - 0.7: Mixed but more boost than downrank
        - 0.3: Neutral (no signals) - should not be favored
        - 0.15: More downrank than boost
        - 0.05: Clear off-domain (downrank terms, no boost) - very harsh

    When NO domain is detected, returns 0.5 (neutral) for all papers.
    """
    if not domain or domain not in DOMAIN_REGISTRY:
        return 0.5  # Neutral if no domain filter active

    config = DOMAIN_REGISTRY[domain]
    text = f"{title} {abstract or ''}".lower()

    # Count boost and downrank term matches
    boost_count = 0
    downrank_count = 0

    for term in config.boost_terms:
        if term in text:
            boost_count += 1

    for term in config.downrank_terms:
        if term in text:
            downrank_count += 1

    # Compute alignment score - VERY aggressive penalties for off-domain papers
    if downrank_count > 0 and boost_count == 0:
        # Clear mismatch - paper is off-domain (e.g., robotics in civil query)
        # Extremely harsh penalty to push these to very bottom
        return 0.05
    elif downrank_count > boost_count:
        # More downrank than boost signals - still quite harsh
        return 0.15
    elif boost_count > 0 and downrank_count == 0:
        # Clear match - paper is on-domain
        return 1.0
    elif boost_count > downrank_count:
        # More boost than downrank - good but not perfect
        return 0.7
    else:
        # Neutral - no strong signals
        # Papers with no domain signals should NOT be favored when domain is active
        return 0.3


def is_off_domain(
    domain: Optional[str],
    title: str,
    abstract: Optional[str] = None,
) -> bool:
    """Check if a paper is clearly off-domain (has downrank terms, no boost terms).

    This is used for hard filtering in specific_topics category to exclude
    papers that are completely irrelevant to the query domain.
    """
    if not domain or domain not in DOMAIN_REGISTRY:
        return False  # No domain filter active

    config = DOMAIN_REGISTRY[domain]
    text = f"{title} {abstract or ''}".lower()

    has_downrank = any(term in text for term in config.downrank_terms)
    has_boost = any(term in text for term in config.boost_terms)

    # Off-domain = has downrank terms AND no boost terms
    return has_downrank and not has_boost


def compute_domain_alignment_batch(
    domain: Optional[str],
    works: Dict[str, any],
    work_ids: List[str],
) -> Dict[str, float]:
    """Compute domain alignment scores for a batch of papers.

    Args:
        domain: Detected domain name (e.g., "civil_engineering") or None
        works: Dict mapping work_id to WorkForMap objects
        work_ids: List of work IDs to score

    Returns:
        Dict mapping work_id to domain alignment score (0.0-1.0)
    """
    result: Dict[str, float] = {}

    for wid in work_ids:
        work = works.get(wid)
        if not work:
            result[wid] = 0.5  # Neutral for missing works
            continue

        title = work.title or ""
        abstract = getattr(work, "abstract", None) or ""

        result[wid] = compute_domain_alignment(domain, title, abstract)

    # Log summary
    if domain:
        high_align = sum(1 for v in result.values() if v >= 0.8)
        low_align = sum(1 for v in result.values() if v <= 0.4)
        logger.info(
            f"Domain alignment ({domain}): {high_align} high, {low_align} low, "
            f"{len(result) - high_align - low_align} neutral out of {len(result)}"
        )

    return result
