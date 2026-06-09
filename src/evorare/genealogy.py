"""Genealogy-collapse statistics from known parent links (pure stdlib, iterative).

The parent links are *known exactly*, so there is nothing to estimate: no probabilistic
ancestry model, no effective-population-size estimate, and no most-recent-common-ancestor
*age* estimation (CHANGE#5; those terms of art are deliberately avoided). We report
two exact descriptors:

  * lineage survivorship ``L(g)`` — the number of distinct founder lineages still represented at
    generation ``g`` (it falls as lineages die out);
  * genealogical mean pairwise distance ``MPD`` — the mean parent-hop distance between the most
    recent individuals, via their lowest common ancestor (it falls as ancestry concentrates).

All ancestor walks are iterative (no recursion) to stay safe on deep genealogies; a generation-
ordered archive is acyclic, but a pathological cycle is broken to guarantee termination (the
resulting depth/LCA values for such malformed input are then unspecified, not meaningful). This
module activates only when ``parent_id`` is present (otherwise it skips, never fabricating links).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import ArchiveRecord


@dataclass(frozen=True)
class GenealogyResult:
    available: bool
    reason: str = ""
    survivorship: tuple[tuple[int, int], ...] = ()  # (generation, distinct_lineages)
    mpd_recent: float = 0.0
    n_roots: int = 0
    max_depth: int = 0


class _Forest:
    """Parent map with memoised iterative root and depth, and iterative LCA."""

    def __init__(self, parent: dict[str, str | None]) -> None:
        self._parent = parent
        self._root: dict[str, str] = {}
        self._depth: dict[str, int] = {}

    def root(self, node: str) -> str:
        if node in self._root:
            return self._root[node]
        path: list[str] = []
        cur: str = node
        seen: set[str] = set()
        while self._parent.get(cur) is not None and cur not in self._root and cur not in seen:
            seen.add(cur)
            path.append(cur)
            nxt = self._parent[cur]
            if nxt is None:
                break
            cur = nxt
        base = self._root.get(cur, cur)
        for p in path:
            self._root[p] = base
        self._root[node] = base
        return base

    def depth(self, node: str) -> int:
        if node in self._depth:
            return self._depth[node]
        path: list[str] = []
        cur: str = node
        seen: set[str] = set()
        d = 0
        while self._parent.get(cur) is not None and cur not in seen:
            if cur in self._depth:
                d = self._depth[cur]
                break
            seen.add(cur)
            path.append(cur)
            nxt = self._parent[cur]
            if nxt is None:
                break
            cur = nxt
        for offset, p in enumerate(reversed(path)):
            self._depth[p] = d + offset + 1
        return self._depth.get(node, 0)

    def ancestors(self, node: str) -> list[str]:
        chain: list[str] = [node]
        cur: str = node
        seen: set[str] = {node}
        while self._parent.get(cur) is not None:
            nxt = self._parent[cur]
            if nxt is None or nxt in seen:
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        return chain

    def lca(self, u: str, v: str) -> str | None:
        au = set(self.ancestors(u))
        for node in self.ancestors(v):
            if node in au:
                return node
        return None

    def pair_distance(self, u: str, v: str) -> int | None:
        anc = self.lca(u, v)
        if anc is None:
            return None
        return (self.depth(u) - self.depth(anc)) + (self.depth(v) - self.depth(anc))


def _build_forest(records: Sequence[ArchiveRecord]) -> _Forest:
    ids = {r.id for r in records}
    parent: dict[str, str | None] = {}
    for r in records:
        p = r.parent_id
        parent[r.id] = p if (p is not None and p in ids) else None
    return _Forest(parent)


def compute_root_map(records: Sequence[ArchiveRecord]) -> dict[str, str]:
    """Map each record id to its founder-lineage root (memoised iterative walk)."""
    forest = _build_forest(records)
    return {r.id: forest.root(r.id) for r in records}


def build_genealogy(
    records: Sequence[ArchiveRecord],
) -> tuple[GenealogyResult, dict[str, str]]:
    """Build the parent forest once; return the GenealogyResult and the id->root map.

    Sharing a single forest avoids walking the parent links twice when both the survivorship
    descriptors and the bootstrap (which needs the root map) are computed for one archive.
    """
    has_parent = any(r.parent_id is not None for r in records)
    if not has_parent:
        return GenealogyResult(available=False, reason="no parent_id present"), {}

    forest = _build_forest(records)
    rootmap = {r.id: forest.root(r.id) for r in records}

    by_gen: dict[int, list[str]] = {}
    for r in records:
        g = r.generation if r.generation is not None else 0
        by_gen.setdefault(g, []).append(r.id)

    survivorship = [(g, len({rootmap[rid] for rid in by_gen[g]})) for g in sorted(by_gen)]
    all_roots = set(rootmap.values())
    max_depth = max((forest.depth(r.id) for r in records), default=0)

    latest_gen = max(by_gen)
    recent = by_gen[latest_gen]
    dists: list[int] = []
    for i in range(len(recent)):
        for j in range(i + 1, len(recent)):
            d = forest.pair_distance(recent[i], recent[j])
            if d is not None:
                dists.append(d)
    mpd = float(sum(dists) / len(dists)) if dists else 0.0

    result = GenealogyResult(
        available=True,
        survivorship=tuple(survivorship),
        mpd_recent=mpd,
        n_roots=len(all_roots),
        max_depth=max_depth,
    )
    return result, rootmap


def compute_genealogy(records: Sequence[ArchiveRecord]) -> GenealogyResult:
    """Compute lineage survivorship and recent genealogical MPD from parent links."""
    return build_genealogy(records)[0]
