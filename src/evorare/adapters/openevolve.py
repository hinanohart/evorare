"""OpenEvolve checkpoint adapter (example).

OpenEvolve persists each program as JSON (id, code, parent_id, generation, iteration, metrics,
island_id). We read every ``*.json`` under ``<checkpoint>/programs`` (or the directory itself)
and project it onto the evorare schema. The score is taken from ``combined_score`` if present,
else the first numeric metric. Missing optional fields are simply omitted (degrade-clean).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pick_score(metrics: dict[str, Any]) -> float | None:
    if "combined_score" in metrics and isinstance(metrics["combined_score"], (int, float)):
        return float(metrics["combined_score"])
    for value in metrics.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def load_openevolve(path: str) -> list[dict[str, Any]]:
    root = Path(path)
    programs_dir = root / "programs"
    search = programs_dir if programs_dir.is_dir() else root
    records: list[dict[str, Any]] = []
    for jf in sorted(search.glob("*.json")):
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(obj, dict) or "id" not in obj or "code" not in obj:
            continue
        metrics = obj.get("metrics") or {}
        rec: dict[str, Any] = {"id": str(obj["id"]), "code": str(obj["code"])}
        score = _pick_score(metrics) if isinstance(metrics, dict) else None
        if score is not None:
            rec["score"] = score
        if obj.get("generation") is not None:
            rec["generation"] = int(obj["generation"])
        if obj.get("parent_id") is not None:
            rec["parent_id"] = str(obj["parent_id"])
        if obj.get("island_id") is not None:
            rec["island_id"] = int(obj["island_id"])
        records.append(rec)
    return records
