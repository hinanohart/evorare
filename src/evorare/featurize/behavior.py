"""Behaviour-descriptor featurizer (primary, score-binned).

Species = a quantised score bin (phenotypic axis). Requires ``score``; under a score-missing
degrade it reports ``applicable`` False and the syntactic axes carry the diagnosis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..schema import ArchiveRecord
from .base import Featurizer, stable_int


class BehaviorFeaturizer(Featurizer):
    """Score-binned phenotypic species. Distance is Euclidean in score space."""

    name = "behavior"

    def __init__(self, ndigits: int = 2) -> None:
        self.ndigits = ndigits

    def applicable(self, rec: ArchiveRecord) -> bool:
        return rec.score is not None

    def featurize(self, rec: ArchiveRecord) -> int:
        if rec.score is None:
            raise ValueError("BehaviorFeaturizer requires a score")
        q = round(float(rec.score), self.ndigits) + 0.0  # + 0.0 collapses -0.0 -> 0.0
        return stable_int(f"behavior:{q:.{self.ndigits}f}")

    def descriptor(self, rec: ArchiveRecord) -> NDArray[np.float64]:
        if rec.score is None:
            raise ValueError("BehaviorFeaturizer requires a score")
        return np.array([float(rec.score)], dtype=np.float64)

    def distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(a - b))
