import numpy as np

from evorare.featurize import (
    AstFeaturizer,
    BehaviorFeaturizer,
    NgramFeaturizer,
    cosine_distance,
    distance_matrix,
    get_featurizer,
    stable_int,
    summarize_sample,
)
from evorare.schema import ArchiveRecord


def _rec(code="def f(x):\n    return x + 1\n", score=0.5, rid="a"):
    return ArchiveRecord(id=rid, code=code, score=score)


def test_stable_int_deterministic():
    assert stable_int("hello") == stable_int("hello")
    assert stable_int("a") != stable_int("b")


def test_behavior_same_bin_same_species():
    f = BehaviorFeaturizer(ndigits=2)
    assert f.featurize(_rec(score=0.831)) == f.featurize(_rec(score=0.834))
    assert f.featurize(_rec(score=0.83)) != f.featurize(_rec(score=0.88))


def test_behavior_negative_zero_same_species_as_zero():
    # -0.0 and 0.0 are the same phenotype; they must not split into two species.
    f = BehaviorFeaturizer(ndigits=2)
    assert f.featurize(_rec(score=-0.0)) == f.featurize(_rec(score=0.0))


def test_behavior_distance_euclidean():
    f = BehaviorFeaturizer()
    a = f.descriptor(_rec(score=0.2))
    b = f.descriptor(_rec(score=0.5))
    assert abs(f.distance(a, b) - 0.3) < 1e-9


def test_behavior_requires_score():
    f = BehaviorFeaturizer()
    assert not f.applicable(ArchiveRecord(id="a", code="x", score=None))


def test_ast_structure_equivalence_ignores_names_constants():
    f = AstFeaturizer()
    s1 = f.featurize(_rec(code="def f(x):\n    return x + 1\n"))
    s2 = f.featurize(_rec(code="def g(y):\n    return y + 99\n"))
    assert s1 == s2  # same shape, different identifiers/constants


def test_ast_different_shape_different_species():
    f = AstFeaturizer()
    s1 = f.featurize(_rec(code="def f(x):\n    return x + 1\n"))
    s2 = f.featurize(_rec(code="def f(x):\n    return (x + 1) * 2\n"))
    assert s1 != s2


def test_ast_not_applicable_on_garbage():
    f = AstFeaturizer()
    assert not f.applicable(_rec(code="def (:::"))


def test_ngram_always_applicable_and_distance_range():
    f = NgramFeaturizer()
    assert f.applicable(_rec(code="not python !!!"))
    a = f.descriptor(_rec(code="hello world"))
    b = f.descriptor(_rec(code="hello there"))
    d = f.distance(a, b)
    assert 0.0 <= d <= 1.0


def test_cosine_distance_identity_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(v, v) < 1e-12


def test_summarize_and_distance_matrix_shapes():
    f = BehaviorFeaturizer()
    recs = [_rec(score=s, rid=str(i)) for i, s in enumerate([0.1, 0.1, 0.5, 0.9])]
    ss = summarize_sample(f, recs)
    assert ss.richness == 3
    assert float(ss.counts.sum()) == 4.0
    mat = distance_matrix(f, ss.representatives)
    assert mat.shape == (3, 3)
    assert np.allclose(mat, mat.T)
    assert np.all(np.diag(mat) == 0.0)


def test_get_featurizer_unknown_raises():
    try:
        get_featurizer("nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
