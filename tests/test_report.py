import json

from evorare.ingest import build_archive
from evorare.judge import diagnose
from evorare.report import diagnosis_to_dict, diagnosis_to_svg, env_stamp, write_json, write_svg
from evorare.synth import generate


def _diag():
    recs, _ = generate("S-HEALTHY", 0)
    arc = build_archive(recs)
    return diagnose(arc, seed=0, n_boot=60), arc


def test_env_stamp_fields():
    e = env_stamp(7)
    for k in ("tool", "version", "python", "numpy", "platform", "date_utc", "seed"):
        assert k in e
    assert e["seed"] == 7
    assert e["tool"] == "evorare"


def test_dict_has_core_keys():
    d, arc = _diag()
    payload = diagnosis_to_dict(d, arc)
    for k in ("env", "verdict", "resolutions", "genealogy", "non_claim", "trend_available"):
        assert k in payload
    assert "does not estimate population diversity" in payload["non_claim"]


def test_dict_is_json_serializable():
    d, arc = _diag()
    payload = diagnosis_to_dict(d, arc)
    s = json.dumps(payload)
    assert len(s) > 100


def test_write_json(tmp_path):
    d, arc = _diag()
    p = tmp_path / "r.json"
    write_json(d, arc, p)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["verdict"] == d.verdict


def test_svg_is_wellformed():
    d, _ = _diag()
    svg = diagnosis_to_svg(d)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "polyline" in svg


def test_write_svg(tmp_path):
    d, _ = _diag()
    p = tmp_path / "r.svg"
    write_svg(d, p)
    assert p.read_text(encoding="utf-8").startswith("<svg")


def test_svg_carries_non_claim():
    d, _ = _diag()
    svg = diagnosis_to_svg(d)
    assert "does not estimate population diversity" in svg
