"""Featurizers: behaviour (primary), AST (co-run), n-gram (fallback).

Default diagnosis uses behaviour + AST as a multi-resolution pair. The n-gram featurizer is
the fallback syntactic axis when the archive is largely non-Python (AST cannot parse it); it
is selected at the featurizer level rather than mixed into a single Rao Q (which requires one
descriptor space). The embedding featurizer is experimental and lives behind the optional
``[embed]`` extra, never in the core import path.
"""

from __future__ import annotations

from .ast_hash import AstFeaturizer
from .base import (
    Featurizer,
    SampleSpecies,
    cosine_distance,
    distance_matrix,
    stable_int,
    summarize_sample,
)
from .behavior import BehaviorFeaturizer
from .ngram import NgramFeaturizer

_REGISTRY: dict[str, type[Featurizer]] = {
    "behavior": BehaviorFeaturizer,
    "ast": AstFeaturizer,
    "ngram": NgramFeaturizer,
}

DEFAULT_FEATURIZERS = ("behavior", "ast")


def get_featurizer(name: str) -> Featurizer:
    """Instantiate a featurizer by name (``behavior`` / ``ast`` / ``ngram``)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown featurizer {name!r}; choose from {sorted(_REGISTRY)}") from None
    return cls()


def available_featurizers() -> list[str]:
    """Names of the registered (non-experimental) featurizers."""
    return sorted(_REGISTRY)


__all__ = [
    "Featurizer",
    "SampleSpecies",
    "BehaviorFeaturizer",
    "AstFeaturizer",
    "NgramFeaturizer",
    "DEFAULT_FEATURIZERS",
    "get_featurizer",
    "available_featurizers",
    "summarize_sample",
    "distance_matrix",
    "cosine_distance",
    "stable_int",
]
