"""Synthetic, labelled archives with known ground-truth diversity behaviour.

These generators are the lifeline of the sensitivity gates (G1-G9): every headline claim is
validated against scenarios whose correct verdict is known by construction. They are CPU-only
and deterministic given a seed.
"""

from __future__ import annotations

from .generators import SCENARIOS, expected_verdict, generate

__all__ = ["SCENARIOS", "generate", "expected_verdict"]
