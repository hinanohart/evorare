"""Labelled synthetic-archive generators (deterministic given a seed).

Each scenario maps a (structure index, score index) per record onto a Python program (whose AST
shape encodes the structure index) and a fitness score (whose value encodes the score index),
plus a lineage assignment with exact parent links. Aligned scenarios move the behaviour and AST
axes together; S-EXPLOIT decouples them; the bottleneck collapses lineages on a planted
generation while leaving diversity healthy.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..schema import ArchiveRecord

N_LINEAGES = 10

SCENARIOS = (
    "S-NULL",
    "S-HEALTHY",
    "S-SATURATE",
    "S-STATIONARY",
    "S-EXPLOIT",
    "S-AGGREGATED",
    "S-BOTTLENECK",
)

_SALT = {name: i + 1 for i, name in enumerate(SCENARIOS)}

_EXPECTED: dict[str, str | None] = {
    "S-NULL": "INDETERMINATE",
    "S-HEALTHY": "HEALTHY",
    "S-SATURATE": "SATURATING",
    "S-STATIONARY": "INDETERMINATE",
    "S-EXPLOIT": "INDETERMINATE",
    "S-AGGREGATED": None,  # routing-gate scenario; verdict is not the assertion (G8)
    "S-BOTTLENECK": "GENEALOGY-COLLAPSE",
}


def expected_verdict(scenario: str) -> str | None:
    return _EXPECTED[scenario]


def _code(structure: int) -> str:
    expr = "x"
    for i in range(max(0, structure)):
        expr = f"({expr} + {i % 7})"
    return f"def f(x):\n    return {expr}\n"


def _score(score_idx: int) -> float:
    return round(0.02 + score_idx * 0.03, 4)


Pair = tuple[int, int]  # (structure_idx, score_idx)


def _b_null(rng: np.random.Generator, n_gen: int, n_per_gen: int) -> list[list[Pair]]:
    s = 15
    out: list[list[Pair]] = []
    for _ in range(n_gen):
        gen = [int(rng.integers(0, s)) for _ in range(n_per_gen)]
        out.append([(i, i) for i in gen])
    return out


def _b_healthy(rng: np.random.Generator, n_gen: int, n_per_gen: int) -> list[list[Pair]]:
    out: list[list[Pair]] = []
    for g in range(n_gen):
        pool = 4 + 3 * g
        gen = [int(rng.integers(0, pool)) for _ in range(n_per_gen)]
        out.append([(i, i) for i in gen])
    return out


def _b_saturate(rng: np.random.Generator, n_gen: int, n_per_gen: int) -> list[list[Pair]]:
    pool = 20
    out: list[list[Pair]] = []
    for g in range(n_gen):
        w = 0.1 + 0.85 * g / max(1, n_gen - 1)
        gen: list[Pair] = []
        for _ in range(n_per_gen):
            idx = 0 if rng.random() < w else int(rng.integers(0, pool))
            gen.append((idx, idx))
        out.append(gen)
    return out


def _b_stationary(rng: np.random.Generator, n_gen: int, n_per_gen: int) -> list[list[Pair]]:
    window = 12
    out: list[list[Pair]] = []
    for g in range(n_gen):
        gen = [int(rng.integers(g, g + window)) for _ in range(n_per_gen)]
        out.append([(i, i) for i in gen])
    return out


def _b_exploit(rng: np.random.Generator, n_gen: int, n_per_gen: int) -> list[list[Pair]]:
    out: list[list[Pair]] = []
    for g in range(n_gen):
        score_pool = 5 + 4 * g  # behaviour diversity grows strongly
        w = 0.05 + 0.9 * g / max(1, n_gen - 1)  # structure concentrates strongly to idx 0
        gen: list[Pair] = []
        for _ in range(n_per_gen):
            score_idx = int(rng.integers(0, score_pool))
            struct_idx = 0 if rng.random() < w else int(rng.integers(0, 18))
            gen.append((struct_idx, score_idx))
        out.append(gen)
    return out


def _b_aggregated(rng: np.random.Generator, n_gen: int, n_per_gen: int) -> list[list[Pair]]:
    out: list[list[Pair]] = []
    for _ in range(n_gen):
        gen: list[Pair] = []
        for _ in range(n_per_gen):
            idx = 0 if rng.random() < 0.9 else int(rng.integers(1, 16))
            gen.append((idx, idx))
        out.append(gen)
    return out


_BUILDERS = {
    "S-NULL": _b_null,
    "S-HEALTHY": _b_healthy,
    "S-SATURATE": _b_saturate,
    "S-STATIONARY": _b_stationary,
    "S-EXPLOIT": _b_exploit,
    "S-AGGREGATED": _b_aggregated,
    "S-BOTTLENECK": _b_healthy,  # healthy diversity; the lineage schedule collapses
}


def _assemble(
    per_gen: list[list[Pair]],
    alive: list[list[int]],
    rng: np.random.Generator,
) -> list[ArchiveRecord]:
    records: list[ArchiveRecord] = []
    prev: dict[int, list[str]] = {}
    for g, specs in enumerate(per_gen):
        al = alive[g]
        cur: dict[int, list[str]] = defaultdict(list)
        for k, (struct_idx, score_idx) in enumerate(specs):
            lin = al[k % len(al)]
            # Link to the previous generation's same-lineage records; within the founding
            # generation, link to the lineage's first record (its single founder). Only the
            # very first record of a lineage is a true founder (parent None), so each lineage
            # contributes exactly one root and L(g) counts surviving lineages.
            candidates = prev.get(lin) or cur.get(lin)
            if candidates:
                parent: str | None = candidates[int(rng.integers(0, len(candidates)))]
            else:
                parent = None
            rec = ArchiveRecord(
                id=f"r{g}_{k}",
                code=_code(struct_idx + 1),
                score=_score(score_idx),
                generation=g,
                parent_id=parent,
                island_id=lin,
            )
            records.append(rec)
            cur[lin].append(rec.id)
        prev = dict(cur)
    return records


def generate(
    scenario: str,
    seed: int,
    n_gen: int = 8,
    n_per_gen: int = 36,
) -> tuple[list[ArchiveRecord], dict[str, object]]:
    """Return (records, meta) for a labelled scenario."""
    if scenario not in _BUILDERS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng([seed, _SALT[scenario]])
    per_gen = _BUILDERS[scenario](rng, n_gen, n_per_gen)

    if scenario == "S-BOTTLENECK":
        planted = n_gen // 2
        alive = [list(range(N_LINEAGES)) if g < planted else [0] for g in range(n_gen)]
    else:
        planted = None
        alive = [list(range(N_LINEAGES)) for _ in range(n_gen)]

    records = _assemble(per_gen, alive, rng)
    meta: dict[str, object] = {
        "scenario": scenario,
        "expected_verdict": _EXPECTED[scenario],
        "planted_bottleneck": planted,
        "has_parent": True,
        "n_gen": n_gen,
        "n_per_gen": n_per_gen,
    }
    return records, meta
