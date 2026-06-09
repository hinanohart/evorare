"""Sensitivity gates G1-G9 (pre-registered, synthetic ground-truth only).

Each gate exercises real code against labelled scenarios and asserts a machine predicate. The
thresholds are not tuned to make the gates pass; they encode the claim. ``evorare gate`` exits 0
only if every gate passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .diversity import hill_number
from .experimental import chao1, chao1_lognormal_ci
from .featurize import get_featurizer, summarize_sample
from .ingest import build_archive
from .judge import (
    HEALTHY,
    INDETERMINATE,
    SATURATING,
    Diagnosis,
    ResolutionDiagnosis,
    diagnose,
)
from .report import diagnosis_to_dict
from .sampling_validity import chi2_ppf, route_estimators
from .schema import Archive, ArchiveRecord
from .synth import generate

GATE_N_BOOT = 120


@dataclass(frozen=True)
class GateResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateReport:
    results: dict[str, GateResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results.values())


def _diag(
    scenario: str, seed: int, n_boot: int = GATE_N_BOOT
) -> tuple[Diagnosis, dict[str, object], Archive, list[ArchiveRecord]]:
    recs, meta = generate(scenario, seed)
    arc = build_archive(recs)
    return diagnose(arc, seed=seed, n_boot=n_boot), meta, arc, recs


def _res(d: Diagnosis, name: str) -> ResolutionDiagnosis:
    for r in d.resolutions:
        if r.featurizer == name:
            return r
    return d.resolutions[0]


def _rate(flags: list[bool]) -> float:
    return float(np.mean([1.0 if f else 0.0 for f in flags])) if flags else 0.0


def g1(seed: int, n_seed: int) -> GateResult:
    """Calibration on S-NULL + closed-form-CI literature cross-check."""
    recs, _ = generate("S-NULL", seed)
    arc = build_archive(recs)
    feat = get_featurizer("behavior")
    identity = True
    for g in sorted({r.generation for r in arc.records if r.generation is not None}):
        ss = summarize_sample(feat, [r for r in arc.records if r.generation == g])
        if hill_number(ss.counts, 0.0) != float(ss.richness):
            identity = False
    table = {1: 3.841, 5: 11.070, 10: 18.307, 20: 31.410}
    chi_ok = True
    errs = []
    for k, v in table.items():
        approx = chi2_ppf(0.95, k)
        rel = abs(approx - v) / v
        errs.append(f"k{k}={rel:.3f}")
        if rel > (0.05 if k == 1 else 0.02):
            chi_ok = False
    est = chao1(20, 10, 5)
    lo, hi = chao1_lognormal_ci(20, 10, 5)
    chao_ok = abs(est - 30.0) < 1e-9 and lo < est < hi and lo >= 20.0
    passed = identity and chi_ok and chao_ok
    return GateResult(
        passed,
        f"hillq0_identity={identity} chi2_WH[{','.join(errs)}]={chi_ok} "
        f"chao1(est={est:.1f},ci=[{lo:.1f},{hi:.1f}])={chao_ok}",
    )


def g2(seed: int, n_seed: int) -> GateResult:
    """Depletion discrimination: HEALTHY vs SATURATE Hill q1 slope separation (AUC>=0.95)."""
    h, s = [], []
    for i in range(n_seed):
        dh, _, _, _ = _diag("S-HEALTHY", seed + i)
        ds, _, _, _ = _diag("S-SATURATE", seed + i)
        h.append(_res(dh, "behavior").hill_q1_slope)
        s.append(_res(ds, "behavior").hill_q1_slope)
    pairs = wins = 0.0
    for a in h:
        for b in s:
            pairs += 1
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    auc = wins / pairs if pairs else 0.0
    passed = auc >= 0.95
    return GateResult(passed, f"AUC={auc:.3f} healthy~{np.mean(h):+.2f} saturate~{np.mean(s):+.2f}")


def g3(seed: int, n_seed: int) -> GateResult:
    """Bottleneck early detection within +-2 generations; low FPR on stationary."""
    within, collapse = [], []
    for i in range(n_seed):
        d, meta, _, _ = _diag("S-BOTTLENECK", seed + i)
        pb = meta["planted_bottleneck"]
        planted = pb if isinstance(pb, int) else 0
        b = d.bottleneck_generation
        within.append(b is not None and abs(b - planted) <= 2)
        collapse.append(d.genealogy_collapse)
    fp = []
    for i in range(n_seed):
        d, _, _, _ = _diag("S-STATIONARY", seed + i)
        fp.append(d.genealogy_collapse)
    det_rate, coll_rate, fpr = _rate(within), _rate(collapse), _rate(fp)
    passed = det_rate >= 0.95 and coll_rate >= 0.95 and fpr < 0.05
    return GateResult(
        passed,
        f"detect_within2={det_rate:.2f} collapse_rate={coll_rate:.2f} stationary_fpr={fpr:.2f}",
    )


def g4(seed: int, n_seed: int) -> GateResult:
    """Turnover confound: S-STATIONARY must not be judged SATURATING."""
    not_sat = []
    for i in range(n_seed):
        d, _, _, _ = _diag("S-STATIONARY", seed + i)
        not_sat.append(d.verdict != SATURATING)
    rate = _rate(not_sat)
    passed = rate >= 0.95
    return GateResult(passed, f"not_saturating_rate={rate:.2f}")


def g5(seed: int, n_seed: int) -> GateResult:
    """Resolution separation: S-EXPLOIT behaviour and AST verdicts differ."""
    differ = []
    for i in range(n_seed):
        d, _, _, _ = _diag("S-EXPLOIT", seed + i)
        differ.append(_res(d, "behavior").verdict != _res(d, "ast").verdict)
    rate = _rate(differ)
    passed = rate >= 0.90
    return GateResult(passed, f"resolution_disagreement_rate={rate:.2f}")


def g6(seed: int, n_seed: int) -> GateResult:
    """Multi-resolution agreement returns INDETERMINATE on the exploit conflict."""
    indet = []
    for i in range(n_seed):
        d, _, _, _ = _diag("S-EXPLOIT", seed + i)
        indet.append(d.verdict == INDETERMINATE)
    rate = _rate(indet)
    passed = rate >= 0.90
    return GateResult(passed, f"exploit_indeterminate_rate={rate:.2f}")


def g7(seed: int, n_seed: int) -> GateResult:
    """Degrade: stripping parent_id skips genealogy and leaves the diversity verdict intact."""
    recs, _ = generate("S-HEALTHY", seed)
    arc = build_archive(recs)
    d_with = diagnose(arc, seed=seed, n_boot=GATE_N_BOOT)
    stripped = [
        ArchiveRecord(
            id=r.id,
            code=r.code,
            score=r.score,
            generation=r.generation,
            parent_id=None,
            island_id=r.island_id,
        )
        for r in recs
    ]
    d_no = diagnose(build_archive(stripped), seed=seed, n_boot=GATE_N_BOOT)
    skipped = not d_no.genealogy.available
    beh_same = _res(d_with, "behavior").verdict == _res(d_no, "behavior").verdict
    diversity_healthy = _res(d_no, "behavior").verdict == HEALTHY
    passed = skipped and beh_same and diversity_healthy
    return GateResult(
        passed,
        f"genealogy_skipped={skipped} behavior_verdict_unchanged={beh_same} "
        f"diversity_verdict={_res(d_no, 'behavior').verdict}",
    )


def g8(seed: int, n_seed: int) -> GateResult:
    """Routing moat: coverage is excluded under S-AGGREGATED, retained under S-NULL."""
    agg, null = [], []
    feat = get_featurizer("behavior")
    for i in range(n_seed):
        recs, _ = generate("S-AGGREGATED", seed + i)
        arc = build_archive(recs)
        last = max(r.generation for r in arc.records if r.generation is not None)
        ss = summarize_sample(feat, [r for r in arc.records if r.generation == last])
        rd = route_estimators(ss.counts)
        agg.append(rd.aggregated and "coverage" in rd.excluded_from_stopping)
        recs2, _ = generate("S-NULL", seed + i)
        arc2 = build_archive(recs2)
        last2 = max(r.generation for r in arc2.records if r.generation is not None)
        ss2 = summarize_sample(feat, [r for r in arc2.records if r.generation == last2])
        null.append(route_estimators(ss2.counts).aggregated)
    agg_rate, null_rate = _rate(agg), _rate(null)
    passed = agg_rate >= 0.95 and null_rate < 0.05
    return GateResult(
        passed, f"aggregated_excluded_rate={agg_rate:.2f} null_false_flag_rate={null_rate:.2f}"
    )


def g9(seed: int, n_seed: int) -> GateResult:
    """Determinism: identical metrics on repeated runs (env timestamp excluded)."""
    recs, _ = generate("S-HEALTHY", seed)
    arc = build_archive(recs)
    j1 = diagnosis_to_dict(diagnose(arc, seed=seed, n_boot=GATE_N_BOOT), arc)
    j2 = diagnosis_to_dict(diagnose(arc, seed=seed, n_boot=GATE_N_BOOT), arc)
    j1["env"].pop("date_utc", None)
    j2["env"].pop("date_utc", None)
    s1 = json.dumps(j1, sort_keys=True)
    s2 = json.dumps(j2, sort_keys=True)
    passed = s1 == s2
    return GateResult(passed, f"byte_identical={passed}")


_GATES = {
    "G1": g1,
    "G2": g2,
    "G3": g3,
    "G4": g4,
    "G5": g5,
    "G6": g6,
    "G7": g7,
    "G8": g8,
    "G9": g9,
}


def run_all_gates(seed: int = 0, n_seed: int = 20) -> GateReport:
    """Run every gate and return the aggregated report."""
    results = {name: fn(seed, n_seed) for name, fn in _GATES.items()}
    return GateReport(results=results)
