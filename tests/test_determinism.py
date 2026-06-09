import json

from evorare.featurize import stable_int
from evorare.ingest import build_archive
from evorare.judge import diagnose
from evorare.report import diagnosis_to_dict
from evorare.synth import generate


def _strip_env(d):
    d = json.loads(json.dumps(d))
    d["env"].pop("date_utc", None)
    return d


def test_diagnose_byte_identical_same_seed():
    recs, _ = generate("S-HEALTHY", 5)
    arc = build_archive(recs)
    a = diagnosis_to_dict(diagnose(arc, seed=5, n_boot=80), arc)
    b = diagnosis_to_dict(diagnose(arc, seed=5, n_boot=80), arc)
    assert json.dumps(_strip_env(a), sort_keys=True) == json.dumps(_strip_env(b), sort_keys=True)


def test_featurizer_hash_stable():
    assert stable_int("evorare") == stable_int("evorare")
    assert stable_int("a-b-c") == stable_int("a-b-c")


def test_different_seed_changes_bootstrap_ci():
    recs, _ = generate("S-HEALTHY", 0)
    arc = build_archive(recs)
    a = diagnose(arc, seed=1, n_boot=80)
    b = diagnose(arc, seed=2, n_boot=80)
    # the plug-in slope is seed-independent, but bootstrap CIs differ across seeds
    ra = a.resolutions[0]
    rb = b.resolutions[0]
    assert ra.hill_q1_slope == rb.hill_q1_slope
    assert ra.hill_q1_slope_ci != rb.hill_q1_slope_ci
