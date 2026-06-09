"""Judgement: the early-stopping verdict from per-generation diversity trends.

Primary signals are the generation trend of the **Hill q1 effective number** and **Rao Q**
(CHANGE#1: coverage is never a sole stopping signal). Slopes carry bootstrap confidence
intervals. A *directional conflict* between resolutions (one HEALTHY while another SATURATING)
yields INDETERMINATE; a merely inconclusive resolution does not veto a decisive one. When
``parent_id`` is present, a significant lineage-survivorship decline yields GENEALOGY-COLLAPSE.
Under an insertion-order generation proxy, no trend is claimed (CHANGE#6) and the verdict is
DESCRIPTIVE.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .diversity import (
    hill_number,
    rao_quadratic_entropy,
    sample_coverage,
)
from .featurize import DEFAULT_FEATURIZERS, distance_matrix, get_featurizer, summarize_sample
from .genealogy import GenealogyResult, build_genealogy
from .sampling_validity import route_estimators
from .schema import Archive, ArchiveRecord

HEALTHY = "HEALTHY"
SATURATING = "SATURATING"
GENEALOGY_COLLAPSE = "GENEALOGY-COLLAPSE"
INDETERMINATE = "INDETERMINATE"
DESCRIPTIVE = "DESCRIPTIVE"


@dataclass(frozen=True)
class GenerationPoint:
    generation: int
    n: int
    richness: int
    hill_q0: float
    hill_q1: float
    hill_q2: float
    rao_q: float
    coverage: float
    aggregated: bool


@dataclass(frozen=True)
class ResolutionDiagnosis:
    featurizer: str
    points: tuple[GenerationPoint, ...]
    hill_q1_slope: float
    hill_q1_slope_ci: tuple[float, float]
    rao_slope: float
    rao_slope_ci: tuple[float, float]
    coverage_valid_fraction: float
    verdict: str


@dataclass(frozen=True)
class Diagnosis:
    verdict: str
    trend_available: bool
    generation_source: str
    seed: int
    resolutions: tuple[ResolutionDiagnosis, ...]
    genealogy: GenealogyResult
    genealogy_collapse: bool
    genealogy_slope_ci: tuple[float, float]
    bottleneck_generation: int | None
    notes: tuple[str, ...]


def _slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    xm = float(x.mean())
    ym = float(y.mean())
    denom = float(np.sum((x - xm) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)


def _ci(values: NDArray[np.float64]) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _resolution_verdict(
    hill_ci: tuple[float, float], rao_ci: tuple[float, float], trend: bool
) -> str:
    if not trend:
        return DESCRIPTIVE
    hill_inc = hill_ci[0] > 0.0
    hill_dec = hill_ci[1] < 0.0
    rao_sig_dec = rao_ci[1] < 0.0  # Rao Q significantly decreasing
    rao_sig_inc = rao_ci[0] > 0.0  # Rao Q significantly increasing
    # Symmetric rule: HEALTHY = Hill q1 rising and Rao Q not significantly falling;
    # SATURATING = Hill q1 falling and Rao Q not significantly rising.
    if hill_inc and not rao_sig_dec:
        return HEALTHY
    if hill_dec and not rao_sig_inc:
        return SATURATING
    return INDETERMINATE


def _diagnose_resolution(
    fname: str,
    by_gen: dict[int, list[ArchiveRecord]],
    gens: list[int],
    rng: np.random.Generator,
    n_boot: int,
    trend: bool,
    alpha: float,
) -> ResolutionDiagnosis:
    feat = get_featurizer(fname)
    points: list[GenerationPoint] = []
    boot_hill: list[NDArray[np.float64]] = []
    boot_rao: list[NDArray[np.float64]] = []

    for g in gens:
        recs = by_gen[g]
        ss = summarize_sample(feat, recs)
        dist = distance_matrix(feat, ss.representatives)
        counts = ss.counts
        n_species = len(ss.species_ids)
        sidx = {s: i for i, s in enumerate(ss.species_ids)}
        labels = np.array(
            [sidx[feat.featurize(r)] for r in recs if feat.applicable(r)], dtype=np.int64
        )
        hq1 = hill_number(counts, 1.0)
        rao = rao_quadratic_entropy(counts, dist) if n_species else 0.0
        routing = route_estimators(counts, alpha)
        points.append(
            GenerationPoint(
                generation=g,
                n=int(counts.sum()),
                richness=ss.richness,
                hill_q0=hill_number(counts, 0.0),
                hill_q1=hq1,
                hill_q2=hill_number(counts, 2.0),
                rao_q=rao,
                coverage=sample_coverage(counts),
                aggregated=routing.aggregated,
            )
        )
        hb = np.empty(n_boot, dtype=np.float64)
        rb = np.empty(n_boot, dtype=np.float64)
        if labels.size > 0 and n_species > 0:
            for b in range(n_boot):
                idx = rng.integers(0, labels.size, labels.size)
                cb = np.bincount(labels[idx], minlength=n_species).astype(np.float64)
                hb[b] = hill_number(cb, 1.0)
                rb[b] = rao_quadratic_entropy(cb, dist)
        else:
            hb[:] = hq1
            rb[:] = rao
        boot_hill.append(hb)
        boot_rao.append(rb)

    x = np.arange(len(gens), dtype=np.float64)
    hill_pts = np.array([p.hill_q1 for p in points], dtype=np.float64)
    rao_pts = np.array([p.rao_q for p in points], dtype=np.float64)
    hill_slope = _slope(x, hill_pts)
    rao_slope = _slope(x, rao_pts)

    if trend and len(gens) >= 3:
        bh = np.vstack(boot_hill).T
        br = np.vstack(boot_rao).T
        hs = np.array([_slope(x, bh[b]) for b in range(bh.shape[0])], dtype=np.float64)
        rs = np.array([_slope(x, br[b]) for b in range(br.shape[0])], dtype=np.float64)
        hill_ci = _ci(hs)
        rao_ci = _ci(rs)
    else:
        hill_ci = (float("nan"), float("nan"))
        rao_ci = (float("nan"), float("nan"))

    verdict = _resolution_verdict(hill_ci, rao_ci, trend and len(gens) >= 3)
    cov_valid = float(np.mean([0.0 if p.aggregated else 1.0 for p in points])) if points else 0.0

    return ResolutionDiagnosis(
        featurizer=fname,
        points=tuple(points),
        hill_q1_slope=hill_slope,
        hill_q1_slope_ci=hill_ci,
        rao_slope=rao_slope,
        rao_slope_ci=rao_ci,
        coverage_valid_fraction=cov_valid,
        verdict=verdict,
    )


def _genealogy_decline(
    rootmap: dict[str, str],
    by_gen: dict[int, list[ArchiveRecord]],
    gens: list[int],
    rng: np.random.Generator,
    n_boot: int,
    threshold: float,
) -> tuple[bool, tuple[float, float], int | None]:
    """Bootstrap the lineage-survivorship slope; return (collapse, slope_ci, bottleneck_gen)."""
    gen_ids = {g: [r.id for r in by_gen[g]] for g in gens}
    lg = np.array([len({rootmap[i] for i in gen_ids[g]}) for g in gens], dtype=np.float64)
    x = np.arange(len(gens), dtype=np.float64)
    if len(gens) < 3:
        return False, (float("nan"), float("nan")), None

    slopes = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        lb = np.empty(len(gens), dtype=np.float64)
        for k, g in enumerate(gens):
            ids = gen_ids[g]
            if ids:
                pick = rng.integers(0, len(ids), len(ids))
                lb[k] = len({rootmap[ids[p]] for p in pick})
            else:
                lb[k] = 0.0
        slopes[b] = _slope(x, lb)
    slope_ci = _ci(slopes)

    peak = float(lg.max()) if lg.size else 0.0
    last = float(lg[-1]) if lg.size else 0.0
    rel_drop = (peak - last) / peak if peak > 0 else 0.0
    collapse = (slope_ci[1] < 0.0) and (rel_drop > threshold)

    bottleneck: int | None = None
    if lg.size >= 2:
        drops = lg[:-1] - lg[1:]
        k = int(np.argmax(drops))
        if drops[k] > 0:
            bottleneck = gens[k + 1]
    return collapse, slope_ci, bottleneck


def diagnose(
    archive: Archive,
    featurizers: tuple[str, ...] = DEFAULT_FEATURIZERS,
    seed: int = 0,
    n_boot: int = 200,
    alpha: float = 0.95,
    collapse_threshold: float = 0.3,
) -> Diagnosis:
    """Diagnose an archive's realized-sample diversity trend and emit a verdict."""
    rng = np.random.default_rng(seed)
    by_gen: dict[int, list[ArchiveRecord]] = defaultdict(list)
    for r in archive.records:
        g = r.generation if r.generation is not None else 0
        by_gen[g].append(r)
    gens = sorted(by_gen)
    trend = archive.has_generation and len(gens) >= 3

    notes: list[str] = []
    if not archive.has_generation:
        notes.append("generation is an insertion-order proxy; no trend/slope is claimed (CHANGE#6)")

    chosen: list[str] = []
    for fname in featurizers:
        if fname == "behavior" and not archive.has_score:
            notes.append("behavior featurizer skipped: archive has no score")
            continue
        chosen.append(fname)
    if not chosen:
        chosen = ["ast"]

    resolutions = tuple(
        _diagnose_resolution(fname, by_gen, gens, rng, n_boot, trend, alpha) for fname in chosen
    )

    if not trend:
        overall = DESCRIPTIVE
    else:
        # Multi-resolution agreement: INDETERMINATE only on an actual directional conflict
        # (one resolution HEALTHY while another SATURATING). A merely inconclusive resolution
        # does not veto a decisive one.
        directional = {r.verdict for r in resolutions if r.verdict in (HEALTHY, SATURATING)}
        overall = directional.pop() if len(directional) == 1 else INDETERMINATE

    genealogy, rootmap = build_genealogy(list(archive.records))
    collapse = False
    gslope_ci: tuple[float, float] = (float("nan"), float("nan"))
    bottleneck: int | None = None
    if genealogy.available:
        collapse, gslope_ci, bottleneck = _genealogy_decline(
            rootmap, by_gen, gens, rng, n_boot, collapse_threshold
        )
        if collapse and trend:
            overall = GENEALOGY_COLLAPSE

    return Diagnosis(
        verdict=overall,
        trend_available=trend,
        generation_source=archive.generation_source,
        seed=seed,
        resolutions=resolutions,
        genealogy=genealogy,
        genealogy_collapse=collapse,
        genealogy_slope_ci=gslope_ci,
        bottleneck_generation=bottleneck,
        notes=tuple(notes),
    )
