"""ShinkaEvolve SQLite adapter (example).

ShinkaEvolve persists its archive in a SQLite database (WAL). We read a ``programs`` table
best-effort, mapping common column names onto the evorare schema. sqlite3 is in the standard
library, so this adds no dependency. Schema drift is tolerated: unknown columns are ignored and
missing optional fields are omitted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_ID_COLS = ("id", "program_id", "uuid")
_CODE_COLS = ("code", "program", "source")
_SCORE_COLS = ("combined_score", "score", "fitness", "metric")
_GEN_COLS = ("generation", "gen", "iteration")
_PARENT_COLS = ("parent_id", "parent")
_ISLAND_COLS = ("island_id", "island")


def _first_present(row: dict[str, Any], cols: tuple[str, ...]) -> Any:
    for c in cols:
        if c in row and row[c] is not None:
            return row[c]
    return None


def load_shinka(path: str) -> list[dict[str, Any]]:
    db = Path(path)
    if not db.exists():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    records: list[dict[str, Any]] = []
    try:
        cur = conn.execute("SELECT * FROM programs")
        for raw in cur.fetchall():
            row = dict(raw)
            rid = _first_present(row, _ID_COLS)
            code = _first_present(row, _CODE_COLS)
            if rid is None or code is None:
                continue
            rec: dict[str, Any] = {"id": str(rid), "code": str(code)}
            score = _first_present(row, _SCORE_COLS)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                rec["score"] = float(score)
            gen = _first_present(row, _GEN_COLS)
            if isinstance(gen, int):
                rec["generation"] = gen
            parent = _first_present(row, _PARENT_COLS)
            if parent is not None:
                rec["parent_id"] = str(parent)
            island = _first_present(row, _ISLAND_COLS)
            if isinstance(island, int):
                rec["island_id"] = island
            records.append(rec)
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()
    return records
