from evorare.ingest import build_archive
from evorare.judge import (
    DESCRIPTIVE,
    GENEALOGY_COLLAPSE,
    HEALTHY,
    SATURATING,
    diagnose,
)
from evorare.schema import ArchiveRecord
from evorare.synth import generate


def _diag(scenario, seed=0, n_boot=100):
    recs, meta = generate(scenario, seed)
    return diagnose(build_archive(recs), seed=seed, n_boot=n_boot), meta


def test_healthy_verdict():
    d, _ = _diag("S-HEALTHY")
    assert d.verdict == HEALTHY
    assert d.trend_available


def test_saturate_verdict():
    d, _ = _diag("S-SATURATE")
    assert d.verdict == SATURATING


def test_bottleneck_collapse():
    d, meta = _diag("S-BOTTLENECK")
    assert d.verdict == GENEALOGY_COLLAPSE
    assert d.genealogy_collapse
    assert abs(d.bottleneck_generation - meta["planted_bottleneck"]) <= 2


def test_stationary_not_saturating():
    d, _ = _diag("S-STATIONARY")
    assert d.verdict != SATURATING


def test_proxy_generation_is_descriptive():
    recs, _ = generate("S-HEALTHY", 0)
    stripped = [
        ArchiveRecord(id=r.id, code=r.code, score=r.score, generation=None, parent_id=r.parent_id)
        for r in recs
    ]
    d = diagnose(build_archive(stripped), seed=0, n_boot=60)
    assert not d.trend_available
    assert d.verdict == DESCRIPTIVE
    assert any("proxy" in n for n in d.notes)


def test_resolutions_present():
    d, _ = _diag("S-HEALTHY")
    names = {r.featurizer for r in d.resolutions}
    assert "behavior" in names
    assert "ast" in names


def test_no_score_drops_behavior():
    recs, _ = generate("S-HEALTHY", 0)
    stripped = [
        ArchiveRecord(
            id=r.id, code=r.code, score=None, generation=r.generation, parent_id=r.parent_id
        )
        for r in recs
    ]
    d = diagnose(build_archive(stripped), seed=0, n_boot=60)
    assert "behavior" not in {r.featurizer for r in d.resolutions}
