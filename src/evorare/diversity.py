"""Per-sample diversity descriptors: Hill numbers, Rao quadratic entropy, sample coverage.

These are *descriptive statistics of the realized sample*. They are well defined regardless of
the sampling design (CHANGE#1 rationale): selection bias distorts the population *interpretation*
of these values, not the values themselves. Population extrapolation (Chao point estimates,
iNEXT) is therefore quarantined in :mod:`evorare.experimental`, not here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hill_number(counts: NDArray[np.float64], q: float) -> float:
    """Hill number (effective number of species) of order ``q`` for a sample.

    q=0 -> observed richness; q=1 -> exp(Shannon); q=2 -> inverse Simpson.
    """
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c / total
    if q == 0:
        return float(p.size)
    if abs(q - 1.0) < 1e-12:
        return float(np.exp(-np.sum(p * np.log(p))))
    return float(np.sum(p**q) ** (1.0 / (1.0 - q)))


def hill_spectrum(
    counts: NDArray[np.float64], q_values: tuple[float, ...] = (0.0, 1.0, 2.0)
) -> dict[float, float]:
    """Hill numbers for several orders at once."""
    return {q: hill_number(counts, q) for q in q_values}


def rao_quadratic_entropy(counts: NDArray[np.float64], dist: NDArray[np.float64]) -> float:
    """Rao quadratic entropy ``Q = sum_ij p_i p_j d_ij``.

    Bounded in ``[0, max(dist)]``. ``dist`` must be a square symmetric distance matrix aligned
    with ``counts``. Cost is O(species^2).
    """
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total <= 0 or c.size == 0:
        return 0.0
    p = c / total
    if dist.shape != (c.size, c.size):
        raise ValueError(f"dist shape {dist.shape} != ({c.size}, {c.size})")
    return float(p @ dist @ p)


def abundance_freq_counts(counts: NDArray[np.float64]) -> tuple[int, int]:
    """Return (f1, f2): the number of singleton and doubleton species in the sample."""
    c = np.asarray(counts, dtype=np.float64)
    f1 = int(np.sum(c == 1))
    f2 = int(np.sum(c == 2))
    return f1, f2


def sample_coverage(counts: NDArray[np.float64]) -> float:
    """Chao-Jost sample-coverage estimate ``Ĉ`` (Good-Turing family).

    AUXILIARY ONLY. Under random independent sampling this estimates the fraction of the
    population represented; under selection/aggregation it over-estimates completeness (its
    Good-Turing assumption fails). The sampling-validity gate decides whether it may inform a
    stopping decision — it is never a sole stopping signal (CHANGE#1).
    """
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    n = c.sum()
    if n <= 0:
        return 0.0
    f1, f2 = abundance_freq_counts(c)
    if f1 == 0:
        return 1.0
    if f2 == 0:
        # Chao correction with f2 -> 0 guarded (Chao & Jost 2012)
        denom = (n - 1.0) * (f1 - 1.0) + 2.0
        return float(1.0 - (f1 / n) * ((n - 1.0) * (f1 - 1.0) / denom))
    denom = (n - 1.0) * f1 + 2.0 * f2
    return float(1.0 - (f1 / n) * ((n - 1.0) * f1 / denom))


def index_of_dispersion(counts: NDArray[np.float64]) -> float:
    """Fisher's index of dispersion statistic ``sum((n_i - n̄)^2)/n̄`` over the species counts.

    Under random independent (Poisson/multinomial) sampling this is ~ chi-square with
    (S-1) degrees of freedom and mean (S-1). Over-dispersion (clumping from selection) inflates
    it; the sampling-validity gate compares it to a chi-square critical value.
    """
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    s = c.size
    if s < 2:
        return 0.0
    mean = c.mean()
    if mean <= 0:
        return 0.0
    return float(np.sum((c - mean) ** 2) / mean)
