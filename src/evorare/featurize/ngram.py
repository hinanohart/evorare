"""Character n-gram featurizer (fallback for non-code / unparseable programs).

Species = the set signature of distinct n-grams; descriptor = a hashed n-gram count vector.
Used only when the AST featurizer cannot parse the program.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from ..schema import ArchiveRecord
from .base import Featurizer, cosine_distance, stable_int


def _bucket(gram: str, dim: int) -> int:
    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % dim


class NgramFeaturizer(Featurizer):
    """Character n-gram species; cosine distance on hashed count vectors."""

    name = "ngram"

    def __init__(self, n: int = 3, dim: int = 256) -> None:
        self.n = n
        self.dim = dim

    def applicable(self, rec: ArchiveRecord) -> bool:
        return True

    def _grams(self, code: str) -> list[str]:
        if len(code) < self.n:
            return [code] if code else [""]
        return [code[i : i + self.n] for i in range(len(code) - self.n + 1)]

    def featurize(self, rec: ArchiveRecord) -> int:
        signature = ",".join(sorted(set(self._grams(rec.code))))
        return stable_int("ngram:" + signature)

    def descriptor(self, rec: ArchiveRecord) -> NDArray[np.float64]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for gram in self._grams(rec.code):
            vec[_bucket(gram, self.dim)] += 1.0
        return vec

    def distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return cosine_distance(a, b)
