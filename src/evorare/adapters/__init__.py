"""Optional framework adapters (examples, not the primary input path).

The primary input is generic JSON-Lines. These adapters convert two concrete framework formats
to that schema as a convenience; they degrade cleanly when the framework's files are absent.
FunSearch ships no stable persisted export, so no FunSearch adapter is provided (honest).
"""

from __future__ import annotations

import json
from pathlib import Path


def convert_to_jsonl(framework: str, path: str, out: str) -> int:
    """Convert a framework archive to JSON-Lines; return the number of records written."""
    if framework == "openevolve":
        from .openevolve import load_openevolve

        records = load_openevolve(path)
    elif framework == "shinka":
        from .shinka import load_shinka

        records = load_shinka(path)
    else:
        raise ValueError(f"unknown framework {framework!r}")

    out_path = Path(out)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return len(records)


__all__ = ["convert_to_jsonl"]
