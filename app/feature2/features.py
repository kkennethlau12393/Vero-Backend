"""
Feature 2 feature computation utilities for production‑grade ranking.

This module implements helper functions to compute and normalise raw
features used in ranking scholarly works.  It includes Bayesian impact
estimation, recency decay, lexical and semantic relevance normalisation,
topic relevance calculation and metadata completeness.  Robust
normalisation functions are reused from the existing rank service.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from datetime import date

from .work_topic_store import WorkForMap


def robust_norm(values: Dict[str, float]) -> Dict[str, float]:
    """Robustly normalise a dictionary of scores to [0, 1] using p05/p95.

    Identical to the normalisation used in the existing rank service.  Values
    below the 5th percentile map to 0 and above the 95th percentile map
    to 1.  If p95 == p05, all outputs are set to 0.
    """
    if not values:
        return {}
    vals = sorted(values.values())
    n = len(vals)
    p05 = vals[max(0, int(round(0.05 * (n - 1))))] if n > 0 else 0.0
    p95 = vals[min(n - 1, int(round(0.95 * (n - 1))))] if n > 0 else 0.0
    if p95 <= p05:
        return {k: 0.0 for k in values.keys()}
    denom = (p95 - p05)
    out: Dict[str, float] = {}
    for k, x in values.items():
        v = (x - p05) / denom
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out[k] = float(v)
    return out


def bayesian_impact_rate(
    cited_by_count: int,
    age_years: float,
    alpha: float = 1.0,
    beta: float = 2.0,
    *,
    uncertainty_gamma: float = 0.0,
    base_rate: Optional[float] = None,
) -> float:
    """Estimate a smoothed citation velocity with optional uncertainty adjustment.

    This function implements the posterior mean of a Gamma–Poisson conjugate
    model, commonly used to estimate citation velocity given a Poisson
    likelihood on citations with a Gamma(α, β) prior on the rate λ.  The
    posterior mean of λ, given ``cited_by_count`` citations over ``age_years``
    years of exposure, is :math:`(α + c) / (β + t)`, where ``t`` is the
    exposure time.  See MAS3301 Bayesian statistics notes for details【791934982385687†L420-L424】.

    To make this more suitable for production use, two optional adjustments
    are provided:

      • ``uncertainty_gamma`` – if > 0, the returned rate is penalised by
        ``gamma × σ`` where σ is the posterior standard deviation.  This
        yields a lower confidence bound on the citation rate and helps
        prevent very young papers with few citations from ranking too
        highly.  A value of 0 reproduces the posterior mean.

      • ``base_rate`` – if provided and positive, the estimated rate is
        divided by this baseline rate.  This can be used to roughly
        normalise citation velocity across fields or venues (e.g. dividing
        by the median citation rate in the paper’s field).  If not
        provided, no normalisation is applied.

    Parameters
    ----------
    cited_by_count : int
        Number of citations received.
    age_years : float
        Age of the work in years.  A value of 0 indicates a very recent work.
    alpha, beta : float
        Parameters of the Gamma prior on the citation rate λ.  Larger
        alpha/beta shrink the posterior towards the prior.  Defaults to 1
        and 2 respectively, which produce a conservative prior rate of
        1/2 citations per year.
    uncertainty_gamma : float, optional
        Number of posterior standard deviations to subtract from the mean
        when computing a risk‑adjusted citation rate.  The default of 0
        returns the posterior mean.  Increasing this value downweights
        uncertain estimates.
    base_rate : float, optional
        A baseline citation rate for the field or venue.  If provided and
        positive, the final estimate will be divided by this value to
        produce a dimensionless normalised rate.

    Returns
    -------
    float
        A smoothed (and optionally normalised) estimate of the citation
        velocity.
    """
    # Ensure non‑negative exposure; add a small epsilon to avoid division by zero.
    exposure = max(age_years, 0.0) + 1.0
    # Posterior mean of the Gamma–Poisson model
    mean_rate = (alpha + cited_by_count) / (beta + exposure)
    # Posterior variance: (α + c) / (β + t)^2
    var_rate = (alpha + cited_by_count) / ((beta + exposure) ** 2)
    # Apply uncertainty penalty if requested
    adjusted_rate = mean_rate - uncertainty_gamma * math.sqrt(var_rate)
    # Normalise by baseline if provided
    if base_rate and base_rate > 0:
        adjusted_rate = adjusted_rate / base_rate
    # Ensure non‑negative result
    return max(0.0, adjusted_rate)


def compute_recency(age_years: float, half_life_years: float) -> float:
    """Compute an exponential decay recency factor.

    A half‑life of `half_life_years` means the recency weight halves every
    `half_life_years` years.  If `age_years` is negative or None, returns 0.
    """
    if age_years is None or age_years < 0:
        return 0.0
    # Prevent division by zero
    hl = half_life_years if half_life_years > 0 else 1.0
    return math.exp(-age_years / hl)


def compute_topic_relevance(w: WorkForMap, target_topic_id: Optional[str]) -> float:
    """Compute a graded relevance score based on topic overlap.

    Returns 1.0 if the work's primary topic matches the target.  Otherwise
    returns the score of the matching topic among its topics list.  If no
    match exists, returns 0.
    """
    if not target_topic_id or not w:
        return 0.0
    if w.primary_topic_id == target_topic_id:
        return 1.0
    best = 0.0
    for t in (w.topics or []):
        if t.topic_id == target_topic_id:
            best = max(best, float(t.score))
    return float(best)


def compute_completeness(w: WorkForMap) -> float:
    """Return a metadata completeness fraction for a work.

    Counts the presence of five metadata fields: title, year, at least one
    author, venue and abstract.  Each missing component reduces the
    completeness score by 0.2.  This function returns a fraction in
    ``[0, 1]``.  Works with no metadata yield 0.
    """
    if not w:
        return 0.0
    total = 5
    present = 0
    if w.title:
        present += 1
    if w.year is not None:
        present += 1
    if isinstance(w.authors_json, list) and len(w.authors_json) > 0:
        present += 1
    if w.venue:
        present += 1
    # Count presence of abstract where available (may be None or empty)
    try:
        # Some WorkForMap implementations may not include abstract; use getattr
        abstract = getattr(w, 'abstract', None)
        if abstract:
            present += 1
    except Exception:
        pass
    return present / float(total)


def compute_age(year: Optional[int]) -> float:
    """Return the age in years given a publication year.

    If the year is None, returns a default age of 10 years to avoid
    penalising missing data too harshly.
    """
    current_year = date.today().year
    if year is None:
        return 10.0
    try:
        return max(0.0, float(current_year - int(year)))
    except Exception:
        return 10.0


def compute_llm_relevance_feature(
    llm_scores: Dict[str, float],
    paper_ids: list[str],
) -> Dict[str, float]:
    """Prepare LLM relevance scores as a feature.

    Papers not in llm_scores get 0.0.
    Returns raw scores; caller should apply robust_norm for normalization.
    """
    result: Dict[str, float] = {}
    for pid in paper_ids:
        result[pid] = llm_scores.get(pid, 0.0)
    return result

