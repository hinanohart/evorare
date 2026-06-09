"""AST canonical-structure featurizer (syntactic axis, co-run with behaviour).

Species = the canonical *shape* of the parsed AST with identifiers and constant values
stripped, so programs that differ only in names/literals share a species. Distance is the
cosine distance between AST node-type histograms. Falls back (``applicable`` False) when the
code does not parse, letting the n-gram featurizer take over.
"""

from __future__ import annotations

import ast
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ..schema import ArchiveRecord
from .base import Featurizer, cosine_distance, stable_int

_NODE_TYPES: list[str] = sorted(
    name
    for name in dir(ast)
    if isinstance(getattr(ast, name), type) and issubclass(getattr(ast, name), ast.AST)
)
_INDEX: dict[str, int] = {name: i for i, name in enumerate(_NODE_TYPES)}


@lru_cache(maxsize=4096)
def _parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def _canon(node: object) -> str:
    """Recursive structural serialisation; primitive leaves (names/constants) are dropped."""
    if isinstance(node, ast.AST):
        parts: list[str] = []
        for field in node._fields:
            val = getattr(node, field, None)
            if isinstance(val, ast.AST):
                parts.append(_canon(val))
            elif isinstance(val, list):
                inner = ",".join(_canon(v) for v in val if isinstance(v, ast.AST))
                parts.append("[" + inner + "]")
        return type(node).__name__ + "(" + ",".join(parts) + ")"
    return ""


def _histogram(tree: ast.AST) -> NDArray[np.float64]:
    vec = np.zeros(len(_NODE_TYPES), dtype=np.float64)
    for node in ast.walk(tree):
        idx = _INDEX.get(type(node).__name__)
        if idx is not None:
            vec[idx] += 1.0
    return vec


class AstFeaturizer(Featurizer):
    """Canonical-AST-shape species; cosine distance on node-type histograms."""

    name = "ast"

    def applicable(self, rec: ArchiveRecord) -> bool:
        return _parse(rec.code) is not None

    def featurize(self, rec: ArchiveRecord) -> int:
        tree = _parse(rec.code)
        if tree is None:
            raise ValueError("AstFeaturizer requires parseable code")
        return stable_int("ast:" + _canon(tree))

    def descriptor(self, rec: ArchiveRecord) -> NDArray[np.float64]:
        tree = _parse(rec.code)
        if tree is None:
            raise ValueError("AstFeaturizer requires parseable code")
        return _histogram(tree)

    def distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return cosine_distance(a, b)
