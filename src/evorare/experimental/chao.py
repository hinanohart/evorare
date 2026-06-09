"""Chao1 richness point estimate and its log-normal confidence interval (Chao 1987).

EXPERIMENTAL. This extrapolates to population richness and assumes near-random sampling. It is
excluded from the default diagnosis and the headline; it exists so gate G1 can cross-check the
self-rolled closed-form CI against the published formula, and as the estimator the
sampling-validity gate routes *out* under selection.
"""

from __future__ import annotations

import math


def chao1(s_obs: int, f1: int, f2: int) -> float:
    """Chao1 estimate of richness. Uses the bias-corrected form when ``f2 == 0``."""
    if f2 > 0:
        return float(s_obs + (f1 * f1) / (2.0 * f2))
    return float(s_obs + (f1 * (f1 - 1)) / 2.0)


def _chao1_variance(s_obs: int, f1: int, f2: int) -> float:
    if f2 > 0:
        r = f1 / f2
        return float(f2 * (0.25 * r**4 + r**3 + 0.5 * r**2))
    # f2 == 0 bias-corrected variance (Chao & Chiu)
    return float(
        0.25 * f1 * (f1 - 1) + 0.5 * f1 * (2 * f1 - 1) ** 2 / 4.0 - (f1**4) / (4.0 * max(1, s_obs))
    )


def chao1_lognormal_ci(
    s_obs: int, f1: int, f2: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Log-normal CI for Chao1 (Chao 1987). ``z`` defaults to the 95% two-sided quantile.

    The interval brackets the point estimate and its lower bound never drops below ``s_obs``.
    """
    est = chao1(s_obs, f1, f2)
    diff = est - s_obs
    if diff <= 0:
        return float(s_obs), float(s_obs)
    var = _chao1_variance(s_obs, f1, f2)
    if var <= 0:
        return float(est), float(est)
    c = math.exp(z * math.sqrt(math.log(1.0 + var / (diff * diff))))
    lo = s_obs + diff / c
    hi = s_obs + diff * c
    return float(lo), float(hi)
