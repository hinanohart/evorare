"""Env-stamped JSON serialisation of a diagnosis (every number is reproducible)."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..judge import Diagnosis
from ..schema import Archive


def env_stamp(seed: int) -> dict[str, Any]:
    return {
        "tool": "evorare",
        "version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
    }


def _ci(ci: tuple[float, float]) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in ci]


def diagnosis_to_dict(diag: Diagnosis, archive: Archive) -> dict[str, Any]:
    resolutions = []
    for res in diag.resolutions:
        resolutions.append(
            {
                "featurizer": res.featurizer,
                "verdict": res.verdict,
                "hill_q1_slope": res.hill_q1_slope,
                "hill_q1_slope_ci": _ci(res.hill_q1_slope_ci),
                "rao_slope": res.rao_slope,
                "rao_slope_ci": _ci(res.rao_slope_ci),
                "coverage_valid_fraction": res.coverage_valid_fraction,
                "per_generation": [
                    {
                        "generation": p.generation,
                        "n": p.n,
                        "richness": p.richness,
                        "hill_q0": p.hill_q0,
                        "hill_q1": p.hill_q1,
                        "hill_q2": p.hill_q2,
                        "rao_q": p.rao_q,
                        "coverage": p.coverage,
                        "coverage_low_confidence": p.aggregated,
                    }
                    for p in res.points
                ],
            }
        )
    genealogy = {
        "available": diag.genealogy.available,
        "reason": diag.genealogy.reason,
        "collapse": diag.genealogy_collapse,
        "survivorship_slope_ci": _ci(diag.genealogy_slope_ci),
        "bottleneck_generation": diag.bottleneck_generation,
        "survivorship": [list(s) for s in diag.genealogy.survivorship],
        "mpd_recent": diag.genealogy.mpd_recent,
        "n_roots": diag.genealogy.n_roots,
        "max_depth": diag.genealogy.max_depth,
    }
    return {
        "env": env_stamp(diag.seed),
        "verdict": diag.verdict,
        "trend_available": diag.trend_available,
        "generation_source": diag.generation_source,
        "archive": {
            "n_records": len(archive),
            "has_score": archive.has_score,
            "has_generation": archive.has_generation,
            "has_parent_id": archive.has_parent_id,
        },
        "non_claim": (
            "Describes realized-sample diversity only; does not estimate population diversity."
        ),
        "resolutions": resolutions,
        "genealogy": genealogy,
        "notes": list(diag.notes),
    }


def write_json(diag: Diagnosis, archive: Archive, path: str | Path) -> dict[str, Any]:
    payload = diagnosis_to_dict(diag, archive)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
