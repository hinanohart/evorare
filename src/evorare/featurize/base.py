"""Featurizer interface and sample-level helpers.

A featurizer maps a program to a *species id* (an integer equivalence-class label) and to a
numeric *descriptor* used by the distance metric that Rao quadratic entropy needs (CHANGE#4:
``distance`` is a required method). Species ids use blake2b, not Python ``hash()``, so they are
stable across processes (a determinism-gate requirement).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ..schema import ArchiveRecord


def stable_int(text: str) -> int:
    """Deterministic 63-bit integer from text via blake2b (no per-process seed)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


class Featurizer(ABC):
    """Maps a program to a species id and a descriptor, with a pairwise distance."""

    name: str

    @abstractmethod
    def applicable(self, rec: ArchiveRecord) -> bool:
        """Whether this featurizer can label ``rec`` (e.g. AST needs parseable code)."""

    @abstractmethod
    def featurize(self, rec: ArchiveRecord) -> int:
        """Return the species id (integer equivalence-class label)."""

    @abstractmethod
    def descriptor(self, rec: ArchiveRecord) -> NDArray[np.float64]:
        """Return the numeric descriptor used by :meth:`distance`."""

    @abstractmethod
    def distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Non-negative distance between two descriptors (CHANGE#4: required)."""


def cosine_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """1 - cosine similarity, clamped to [0, 1]. Zero vectors are distance 0 to themselves."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0 if na == nb else 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return float(min(1.0, max(0.0, 1.0 - sim)))


class SampleSpecies:
    """Species present in one sample under a featurizer: ids, abundances, representatives."""

    def __init__(
        self,
        species_ids: list[int],
        counts: NDArray[np.float64],
        representatives: list[NDArray[np.float64]],
        n_unassigned: int,
    ) -> None:
        self.species_ids = species_ids
        self.counts = counts
        self.representatives = representatives
        self.n_unassigned = n_unassigned

    @property
    def richness(self) -> int:
        return len(self.species_ids)


def summarize_sample(feat: Featurizer, records: Iterable[ArchiveRecord]) -> SampleSpecies:
    """Group records into species; the representative is the mean member descriptor.

    Records the featurizer cannot label (``applicable`` False) are counted as ``n_unassigned``
    and excluded — never fabricated into a species.
    """
    groups: dict[int, list[NDArray[np.float64]]] = {}
    n_unassigned = 0
    for r in records:
        if not feat.applicable(r):
            n_unassigned += 1
            continue
        sid = feat.featurize(r)
        groups.setdefault(sid, []).append(feat.descriptor(r))
    species = sorted(groups)
    counts = np.array([len(groups[s]) for s in species], dtype=np.float64)
    reps = [np.mean(np.stack(groups[s]), axis=0) for s in species]
    return SampleSpecies(species, counts, reps, n_unassigned)


def distance_matrix(feat: Featurizer, reps: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Symmetric pairwise distance matrix over species representatives (zero diagonal)."""
    n = len(reps)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = feat.distance(reps[i], reps[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat
