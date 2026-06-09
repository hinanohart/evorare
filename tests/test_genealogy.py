from evorare.genealogy import compute_genealogy, compute_root_map
from evorare.schema import ArchiveRecord


def _r(rid, parent, gen):
    return ArchiveRecord(id=rid, code="x", score=0.1, generation=gen, parent_id=parent)


def test_no_parent_skips():
    recs = [ArchiveRecord(id=str(i), code="x", score=0.1, generation=0) for i in range(3)]
    res = compute_genealogy(recs)
    assert not res.available
    assert "no parent_id" in res.reason


def test_root_map_chain():
    recs = [_r("a", None, 0), _r("b", "a", 1), _r("c", "b", 2), _r("d", "a", 1)]
    rm = compute_root_map(recs)
    assert rm["a"] == "a"
    assert rm["b"] == "a"
    assert rm["c"] == "a"
    assert rm["d"] == "a"


def test_two_founders_two_roots():
    recs = [_r("a", None, 0), _r("b", None, 0), _r("a2", "a", 1), _r("b2", "b", 1)]
    res = compute_genealogy(recs)
    assert res.available
    assert res.n_roots == 2


def test_survivorship_counts_lineages_per_gen():
    recs = [
        _r("a", None, 0),
        _r("b", None, 0),
        _r("a1", "a", 1),
        _r("b1", "b", 1),
        _r("a2", "a1", 2),  # lineage b dies at gen 2
    ]
    res = compute_genealogy(recs)
    surv = dict(res.survivorship)
    assert surv[0] == 2
    assert surv[1] == 2
    assert surv[2] == 1


def test_max_depth():
    recs = [_r("a", None, 0), _r("b", "a", 1), _r("c", "b", 2)]
    res = compute_genealogy(recs)
    assert res.max_depth == 2


def test_cycle_is_guarded():
    # pathological self/loop should not hang and should still return
    recs = [_r("a", "b", 0), _r("b", "a", 1), _r("c", "a", 1)]
    res = compute_genealogy(recs)
    assert res.available


def test_mpd_recent_nonneg():
    recs = [_r("a", None, 0), _r("a1", "a", 1), _r("a2", "a", 1)]
    res = compute_genealogy(recs)
    assert res.mpd_recent >= 0.0
