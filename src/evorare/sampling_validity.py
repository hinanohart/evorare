"""Sampling-validity routing gate — the moat (CHANGE#3).

Coverage / Chao estimators assume random independent sampling. Evolutionary search is
selection-biased: a few high-fitness species are resampled repeatedly, which is statistically
equivalent to spatial aggregation and makes those estimators over-state completeness. This gate
detects that regime with Fisher's index-of-dispersion test and *routes it out* of the stopping
decision (a fail-closed action, not merely a warning). The threshold is a chi-square critical
value, not a hand-tuned constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .diversity import index_of_dispersion

# Estimators whose validity depends on the random-sampling assumption.
ASSUMPTION_DEPENDENT = ("coverage", "chao1", "chao2", "inext")


def normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, numpy-free)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def chi2_ppf(p: float, k: int) -> float:
    """Chi-square quantile via the Wilson-Hilferty approximation (numpy/scipy-free).

    Accurate to a few percent for k>=1 and well under 1% for k>=5; validated against tabulated
    critical values in the test suite (the closed-form-CI literature cross-check, CHANGE#8).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    z = normal_ppf(p)
    t = 1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))
    return float(k * t**3)


@dataclass(frozen=True)
class RoutingDecision:
    """Outcome of the sampling-validity gate for one sample."""

    aggregated: bool
    dispersion: float
    critical: float
    richness: int
    valid_for_stopping: tuple[str, ...]
    excluded_from_stopping: tuple[str, ...]
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def confidence(self) -> str:
        return "LOW_CONFIDENCE" if self.aggregated else "OK"


def route_estimators(counts: NDArray[np.float64], alpha: float = 0.95) -> RoutingDecision:
    """Decide which estimators may inform the stopping decision for this sample.

    Always-valid (descriptive, assumption-free): Hill numbers and Rao Q.
    Assumption-dependent (coverage/Chao/iNEXT): excluded when over-dispersion exceeds the
    chi-square(alpha, S-1) critical value.
    """
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    richness = int(c.size)
    always_valid = ("hill", "rao_q")
    if richness < 2:
        # cannot run the dispersion test; be conservative and exclude assumption-dependent ones.
        return RoutingDecision(
            aggregated=True,
            dispersion=0.0,
            critical=0.0,
            richness=richness,
            valid_for_stopping=always_valid,
            excluded_from_stopping=ASSUMPTION_DEPENDENT,
            flags=("RICHNESS_LT_2",),
        )
    dispersion = index_of_dispersion(c)
    critical = chi2_ppf(alpha, richness - 1)
    aggregated = dispersion > critical
    if aggregated:
        return RoutingDecision(
            aggregated=True,
            dispersion=dispersion,
            critical=critical,
            richness=richness,
            valid_for_stopping=always_valid,
            excluded_from_stopping=ASSUMPTION_DEPENDENT,
            flags=("OVERDISPERSED_SELECTION",),
        )
    return RoutingDecision(
        aggregated=False,
        dispersion=dispersion,
        critical=critical,
        richness=richness,
        valid_for_stopping=always_valid + ASSUMPTION_DEPENDENT,
        excluded_from_stopping=(),
        flags=(),
    )
