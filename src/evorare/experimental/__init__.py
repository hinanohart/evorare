"""Experimental, default-OFF estimators — NOT part of the v0.1 claim.

These extrapolate to the *population* (Chao1/Chao2 point estimates, iNEXT extrapolation) and so
assume near-random sampling, which evolutionary archives violate. They are kept here, off the
default path and outside the headline, used only as a sampling-validity reference and for the
closed-form-CI literature cross-check in gate G1.
"""

from __future__ import annotations

from .chao import chao1, chao1_lognormal_ci

__all__ = ["chao1", "chao1_lognormal_ci"]
