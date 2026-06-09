import math

import numpy as np

from evorare.sampling_validity import (
    ASSUMPTION_DEPENDENT,
    chi2_ppf,
    normal_ppf,
    route_estimators,
)


def test_normal_ppf_known_values():
    assert abs(normal_ppf(0.975) - 1.959963984540054) < 1e-6
    assert abs(normal_ppf(0.5)) < 1e-9
    assert abs(normal_ppf(0.95) - 1.6448536269514722) < 1e-6


def test_chi2_ppf_matches_table():
    # tabulated chi-square 0.95 critical values
    table = {1: 3.841, 5: 11.070, 10: 18.307, 20: 31.410}
    for k, v in table.items():
        approx = chi2_ppf(0.95, k)
        rel = abs(approx - v) / v
        assert rel < (0.05 if k == 1 else 0.02), (k, approx, v)


def test_routing_uniform_not_aggregated():
    counts = np.array([10.0, 11.0, 9.0, 10.0, 10.0, 11.0, 9.0, 10.0])
    rd = route_estimators(counts)
    assert not rd.aggregated
    assert rd.confidence == "OK"
    assert set(ASSUMPTION_DEPENDENT).issubset(set(rd.valid_for_stopping))


def test_routing_clumped_aggregated_excludes_coverage():
    counts = np.array([200.0, 1.0, 1.0, 1.0, 1.0])
    rd = route_estimators(counts)
    assert rd.aggregated
    assert rd.confidence == "LOW_CONFIDENCE"
    assert "coverage" in rd.excluded_from_stopping
    assert "hill" in rd.valid_for_stopping
    assert "rao_q" in rd.valid_for_stopping


def test_routing_richness_lt_2_conservative():
    rd = route_estimators(np.array([5.0]))
    assert rd.aggregated
    assert "RICHNESS_LT_2" in rd.flags


def test_chi2_increases_with_df():
    assert chi2_ppf(0.95, 1) < chi2_ppf(0.95, 10) < chi2_ppf(0.95, 50)


def test_normal_ppf_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.1):
        try:
            normal_ppf(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_hill_always_valid_regardless():
    # even uniform sample keeps hill/rao always valid for stopping
    rd = route_estimators(np.array([3.0, 3.0, 3.0]))
    assert "hill" in rd.valid_for_stopping
    assert math.isfinite(rd.dispersion)
