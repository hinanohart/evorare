import pytest

from evorare.ingest import build_archive
from evorare.synth import SCENARIOS, expected_verdict, generate


def test_all_scenarios_generate():
    for sc in SCENARIOS:
        recs, meta = generate(sc, 0)
        assert len(recs) > 0
        assert meta["scenario"] == sc


def test_generation_deterministic():
    a, _ = generate("S-HEALTHY", 3)
    b, _ = generate("S-HEALTHY", 3)
    assert [(r.id, r.code, r.score, r.parent_id) for r in a] == [
        (r.id, r.code, r.score, r.parent_id) for r in b
    ]


def test_seeds_differ():
    a, _ = generate("S-HEALTHY", 1)
    b, _ = generate("S-HEALTHY", 2)
    assert [r.score for r in a] != [r.score for r in b]


def test_records_have_parent_and_generation():
    recs, _ = generate("S-BOTTLENECK", 0)
    arc = build_archive(recs)
    assert arc.has_parent_id
    assert arc.has_generation


def test_bottleneck_planted_midway():
    _, meta = generate("S-BOTTLENECK", 0, n_gen=8)
    assert meta["planted_bottleneck"] == 4


def test_codes_parse_as_python():
    import ast

    recs, _ = generate("S-HEALTHY", 0)
    for r in recs[:20]:
        ast.parse(r.code)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        generate("S-NOPE", 0)


def test_expected_verdict_lookup():
    assert expected_verdict("S-HEALTHY") == "HEALTHY"
    assert expected_verdict("S-AGGREGATED") is None
