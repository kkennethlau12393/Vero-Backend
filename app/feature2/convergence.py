"""
Discovery Curve Convergence System.

Fits a Weibull CDF model to cumulative LLM relevance scores collected
in waves, predicting total relevant material and computing completeness.

Model: S(n) = N * (1 - exp(-(n/tau)^k))

Where:
    n   = papers scored so far
    S(n)= cumulative relevance found
    N   = predicted total relevance (asymptote)
    tau = scale parameter
    k   = shape parameter (>1 means front-loaded discovery)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

WAVE_SIZE = 45
MAX_WAVES = 4
COMPLETENESS_TARGET = 0.50
MIN_WAVES_FOR_FIT = 2


@dataclass
class WaveResult:
    papers_scored: int          # cumulative
    cumulative_relevance: float # sum of all LLM scores so far


@dataclass
class ConvergenceState:
    waves: list[WaveResult]
    N: float = 0.0
    tau: float = 0.0
    k: float = 1.0
    completeness: float = 0.0
    confidence: float = 0.0     # R² goodness of fit
    should_continue: bool = True


def _weibull_cdf(n: float, tau: float, k: float) -> float:
    """Compute 1 - exp(-(n/tau)^k)."""
    return 1.0 - math.exp(-((n / tau) ** k))


def _compute_N_closed_form(
    data: list[WaveResult], tau: float, k: float
) -> float:
    """Analytically compute optimal N for given tau, k.

    N = sum(S_i * f_i) / sum(f_i^2)
    where f_i = 1 - exp(-(n_i/tau)^k)
    """
    num = 0.0
    den = 0.0
    for w in data:
        f = _weibull_cdf(w.papers_scored, tau, k)
        num += w.cumulative_relevance * f
        den += f * f
    if den < 1e-12:
        return 0.0
    return num / den


def _residual_sum_of_squares(
    data: list[WaveResult], N: float, tau: float, k: float
) -> float:
    rss = 0.0
    for w in data:
        predicted = N * _weibull_cdf(w.papers_scored, tau, k)
        rss += (w.cumulative_relevance - predicted) ** 2
    return rss


def _compute_r_squared(data: list[WaveResult], N: float, tau: float, k: float) -> float:
    """Compute R² (coefficient of determination) for the fit."""
    if len(data) < 2:
        return 0.0
    mean_s = sum(w.cumulative_relevance for w in data) / len(data)
    ss_tot = sum((w.cumulative_relevance - mean_s) ** 2 for w in data)
    if ss_tot < 1e-12:
        return 1.0  # all points identical = perfect fit
    ss_res = _residual_sum_of_squares(data, N, tau, k)
    return 1.0 - ss_res / ss_tot


def fit_curve(data: list[WaveResult]) -> tuple[float, float, float]:
    """Fit the Weibull CDF model to wave data.

    Returns (N, tau, k).

    With 2 data points: fixes k=1, fits N and tau.
    With 3+ data points: fits all three parameters.
    """
    if len(data) < MIN_WAVES_FOR_FIT:
        # Not enough data; return rough estimate
        last = data[-1] if data else WaveResult(0, 0.0)
        return (last.cumulative_relevance * 1.5, float(last.papers_scored), 1.0)

    max_n = max(w.papers_scored for w in data)

    last_cum = data[-1].cumulative_relevance

    # Hard upper bound on N: can't be more than 10× observed cumulative.
    # This is a physical constraint — prevents degenerate fits.
    N_upper = last_cum * 10.0

    if len(data) == 2:
        # Fix k=1, bounded optimize over tau
        def objective_k1(params):
            tau = params[0]
            N = _compute_N_closed_form(data, tau, 1.0)
            if N <= 0 or N > N_upper:
                return 1e12
            return _residual_sum_of_squares(data, N, tau, 1.0)

        best_cost = 1e12
        best_tau = max_n * 0.5
        tau_bounds = [(1.0, max_n * 5.0)]
        for x0 in [max_n * 0.3, max_n * 0.6, max_n * 1.0, max_n * 2.0]:
            try:
                result = minimize(
                    objective_k1, x0=[x0], method="L-BFGS-B",
                    bounds=tau_bounds, options={"maxiter": 200},
                )
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_tau = result.x[0]
            except Exception:
                pass

        tau = max(best_tau, 1.0)
        N = _compute_N_closed_form(data, tau, 1.0)
        N = max(min(N, N_upper), last_cum)
        return (N, tau, 1.0)

    # 3+ points: bounded multi-start optimize over (tau, k)
    def objective(params):
        tau, k = params
        N = _compute_N_closed_form(data, tau, k)
        if N <= 0 or N > N_upper:
            return 1e12
        return _residual_sum_of_squares(data, N, tau, k)

    best_cost = 1e12
    best_params = (max_n * 0.5, 1.2)
    bounds = [(1.0, max_n * 5.0), (0.1, 5.0)]
    for tau0 in [max_n * 0.3, max_n * 0.7, max_n * 1.5]:
        for k0 in [0.8, 1.2, 1.8]:
            try:
                result = minimize(
                    objective, x0=[tau0, k0], method="L-BFGS-B",
                    bounds=bounds, options={"maxiter": 500},
                )
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_params = (result.x[0], result.x[1])
            except Exception:
                pass

    tau = max(best_params[0], 1.0)
    k = max(best_params[1], 0.1)
    N = _compute_N_closed_form(data, tau, k)
    N = max(min(N, N_upper), last_cum)
    return (N, tau, k)


def _rate_based_N_estimate(waves: list[WaveResult]) -> float | None:
    """Estimate N from the deceleration of marginal gains between waves.

    If wave gains are g1, g2, g3... and each successive gain shrinks by
    a ratio r (g2 = g1*r, g3 = g2*r, ...), then the total sum of an
    infinite geometric series is g1 / (1 - r). Add the already-observed
    cumulative to get N.

    Returns None if gains aren't decelerating (can't estimate).
    """
    if len(waves) < 2:
        return None

    gains = []
    for i in range(1, len(waves)):
        g = waves[i].cumulative_relevance - waves[i - 1].cumulative_relevance
        gains.append(max(g, 0.001))  # floor to avoid division by zero

    # Compute decay ratios between successive gains
    ratios = []
    for i in range(1, len(gains)):
        ratios.append(gains[i] / gains[i - 1])

    if not ratios:
        # Only 2 waves = 1 gain, use the single ratio from wave1 to wave2
        # Assume gain from "wave 0" (first wave) to wave 1 is the first gain
        g0 = waves[0].cumulative_relevance  # first wave's total = its gain
        g1 = gains[0]
        if g0 > 0.001:
            r = g1 / g0
        else:
            return None
    else:
        r = sum(ratios) / len(ratios)

    if r >= 1.0 or r < 0.0:
        return None  # gains not decelerating

    # Dampen the ratio — geometric series is hypersensitive near r=1.
    # Apply a concave dampening: effective_r = r^2 for strong deceleration,
    # which makes the estimate conservative when deceleration is weak.
    effective_r = r * r

    last_gain = gains[-1]
    remaining = last_gain * effective_r / (1.0 - effective_r)
    return waves[-1].cumulative_relevance + remaining


def evaluate_convergence(waves: list[WaveResult]) -> ConvergenceState:
    """Fit curve and decide whether to continue scoring."""
    if len(waves) < MIN_WAVES_FOR_FIT:
        return ConvergenceState(
            waves=waves,
            should_continue=True,
        )

    N, tau, k = fit_curve(waves)
    current = waves[-1].cumulative_relevance

    # Handle edge case: N less than what we already found (bad fit)
    if N < current:
        N = current * 1.1  # bump estimate slightly above observed

    raw_confidence = _compute_r_squared(waves, N, tau, k)
    confidence = max(0.0, min(1.0, raw_confidence))  # clamp to [0, 1]

    # When curve fit is unreliable, blend with rate-based estimate
    rate_N = _rate_based_N_estimate(waves)
    if rate_N is not None and rate_N >= current:
        if confidence < 0.5:
            # Very poor fit — trust rate estimate almost entirely
            N = rate_N
        else:
            ratio = N / rate_N if rate_N > 0 else 1.0
            if ratio > 2.0 or ratio < 0.5 or confidence < 0.95:
                blend_weight = max(0.3, 1.0 - confidence)
                N = (1.0 - blend_weight) * N + blend_weight * rate_N
    elif confidence < 0.5:
        # Poor fit and no rate estimate — be conservative
        N = current * 1.3

    if N < current:
        N = current * 1.01

    completeness = current / N if N > 0 else 0.0
    confidence = max(0.0, min(1.0, _compute_r_squared(waves, N, tau, k)))

    # Simple convergence: stop if target reached OR max waves hit
    should_continue = (
        completeness < COMPLETENESS_TARGET
        and len(waves) < MAX_WAVES
    )

    state = ConvergenceState(
        waves=waves,
        N=N,
        tau=tau,
        k=k,
        completeness=min(completeness, 1.0),
        confidence=confidence,
        should_continue=should_continue,
    )

    logger.info(
        f"Convergence wave {len(waves)}: "
        f"completeness={state.completeness:.2%}, N={N:.1f}, "
        f"tau={tau:.1f}, k={k:.2f}, R²={confidence:.3f}, "
        f"continue={should_continue}"
    )

    return state
