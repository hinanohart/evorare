from evorare.experimental import chao1, chao1_lognormal_ci


def test_chao1_closed_form():
    # S_obs=20, f1=10, f2=5 -> 20 + 100/10 = 30
    assert chao1(20, 10, 5) == 30.0


def test_chao1_f2_zero_bias_corrected():
    # f2=0 -> S_obs + f1*(f1-1)/2
    assert chao1(10, 4, 0) == 10 + 4 * 3 / 2


def test_chao1_ci_brackets_estimate():
    est = chao1(20, 10, 5)
    lo, hi = chao1_lognormal_ci(20, 10, 5)
    assert lo < est < hi
    assert lo >= 20.0


def test_chao1_no_extrapolation_when_complete():
    # no singletons -> estimate == observed -> CI degenerate at S_obs
    lo, hi = chao1_lognormal_ci(15, 0, 3)
    assert lo == 15.0 and hi == 15.0


def test_chao1_f2_zero_ci_branch_brackets_estimate():
    # f2 == 0 bias-corrected branch must still produce an ordered CI bracketing the estimate
    est = chao1(12, 6, 0)
    lo, hi = chao1_lognormal_ci(12, 6, 0)
    assert lo < est < hi
    assert lo >= 12.0
