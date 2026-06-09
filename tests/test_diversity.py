import math

import numpy as np

from evorare.diversity import (
    hill_number,
    index_of_dispersion,
    rao_quadratic_entropy,
    sample_coverage,
)


def test_hill_q0_is_richness():
    counts = np.array([5.0, 3.0, 1.0, 0.0])
    assert hill_number(counts, 0.0) == 3.0


def test_hill_uniform_equals_richness():
    counts = np.array([5.0, 5.0, 5.0, 5.0])
    assert math.isclose(hill_number(counts, 1.0), 4.0, rel_tol=1e-9)
    assert math.isclose(hill_number(counts, 2.0), 4.0, rel_tol=1e-9)


def test_hill_q2_inverse_simpson():
    counts = np.array([3.0, 1.0])
    # 1 / (0.75^2 + 0.25^2) = 1/0.625 = 1.6
    assert math.isclose(hill_number(counts, 2.0), 1.6, rel_tol=1e-9)


def test_hill_q1_shannon():
    counts = np.array([3.0, 1.0])
    p = np.array([0.75, 0.25])
    expected = math.exp(-float(np.sum(p * np.log(p))))
    assert math.isclose(hill_number(counts, 1.0), expected, rel_tol=1e-9)


def test_hill_ordering():
    counts = np.array([10.0, 5.0, 2.0, 1.0])
    q0 = hill_number(counts, 0.0)
    q1 = hill_number(counts, 1.0)
    q2 = hill_number(counts, 2.0)
    assert q0 >= q1 >= q2


def test_rao_bounds():
    counts = np.array([1.0, 1.0])
    dist = np.array([[0.0, 0.4], [0.4, 0.0]])
    # p=[.5,.5] -> Q = 2*.5*.5*.4 = 0.2
    assert math.isclose(rao_quadratic_entropy(counts, dist), 0.2, rel_tol=1e-9)


def test_rao_zero_for_single_species():
    counts = np.array([7.0])
    dist = np.zeros((1, 1))
    assert rao_quadratic_entropy(counts, dist) == 0.0


def test_rao_in_range():
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10, size=5).astype(float)
    dmat = rng.random((5, 5))
    dmat = (dmat + dmat.T) / 2
    np.fill_diagonal(dmat, 0.0)
    q = rao_quadratic_entropy(counts, dmat)
    assert 0.0 <= q <= float(dmat.max()) + 1e-9


def test_coverage_singletons_low():
    many_singletons = np.array([1.0] * 10)
    cov = sample_coverage(many_singletons)
    assert cov < 0.5


def test_coverage_no_singletons_full():
    counts = np.array([5.0, 5.0, 5.0])
    assert sample_coverage(counts) == 1.0


def test_index_of_dispersion_uniform_low():
    counts = np.array([10.0, 10.0, 10.0, 10.0])
    assert index_of_dispersion(counts) == 0.0


def test_index_of_dispersion_clumped_high():
    counts = np.array([100.0, 1.0, 1.0, 1.0])
    assert index_of_dispersion(counts) > 100.0
