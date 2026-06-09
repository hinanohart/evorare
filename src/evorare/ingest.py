"""JSON-Lines ingest with a minimal contract validator and fabrication-free degrade.

Degrade rules (machine, no fabrication):
  * ``parent_id`` absent for every record -> genealogy module is skipped.
  * ``generation`` absent for every record -> insertion-order proxy with a tag; under the
    proxy, trend/slope claims are suppressed and only cumulative spectra are reported.
  * ``score`` absent -> behaviour featurizer is disabled (syntactic axes only).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import Archive, ArchiveRecord


class ContractError(ValueError):
    """Raised when a record violates the hard minimal contract (missing id/code)."""


def _coerce_record(obj: dict[str, object], lineno: int) -> ArchiveRecord:
    if "id" not in obj or "code" not in obj:
        raise ContractError(f"line {lineno}: record requires 'id' and 'code'")
    rid = obj["id"]
    code = obj["code"]
    if not isinstance(rid, (str, int)):
        raise ContractError(f"line {lineno}: 'id' must be str/int")
    if not isinstance(code, str):
        raise ContractError(f"line {lineno}: 'code' must be a string")

    score_raw = obj.get("score")
    score: float | None = None
    if score_raw is not None:
        if isinstance(score_raw, bool) or not isinstance(score_raw, (int, float)):
            raise ContractError(f"line {lineno}: 'score' must be a number")
        score = float(score_raw)
        if not math.isfinite(score):
            raise ContractError(f"line {lineno}: 'score' must be finite (got {score_raw!r})")

    gen_raw = obj.get("generation")
    generation: int | None = None
    if gen_raw is not None:
        if isinstance(gen_raw, bool) or not isinstance(gen_raw, int):
            raise ContractError(f"line {lineno}: 'generation' must be an integer")
        generation = int(gen_raw)

    parent_raw = obj.get("parent_id")
    parent_id: str | None = None
    if parent_raw is not None:
        parent_id = str(parent_raw)

    island_raw = obj.get("island_id")
    island_id: int | None = None
    if island_raw is not None:
        if isinstance(island_raw, bool) or not isinstance(island_raw, int):
            raise ContractError(f"line {lineno}: 'island_id' must be an integer")
        island_id = int(island_raw)

    return ArchiveRecord(
        id=str(rid),
        code=code,
        score=score,
        generation=generation,
        parent_id=parent_id,
        island_id=island_id,
    )


def build_archive(records: Iterable[ArchiveRecord]) -> Archive:
    """Assemble an :class:`Archive` from records, applying degrade rules.

    When generations are missing, an insertion-order proxy is filled in and tagged.
    """
    recs = list(records)
    has_score = all(r.score is not None for r in recs) and len(recs) > 0
    has_parent_id = any(r.parent_id is not None for r in recs)
    explicit_gen = all(r.generation is not None for r in recs) and len(recs) > 0

    if explicit_gen:
        generation_source = "explicit"
        final = tuple(recs)
    else:
        generation_source = "insertion-order(proxy)"
        final = tuple(
            ArchiveRecord(
                id=r.id,
                code=r.code,
                score=r.score,
                generation=i,
                parent_id=r.parent_id,
                island_id=r.island_id,
            )
            for i, r in enumerate(recs)
        )

    return Archive(
        records=final,
        has_score=has_score,
        has_generation=explicit_gen,
        has_parent_id=has_parent_id,
        generation_source=generation_source,
    )


def iter_jsonl(path: str | Path) -> Iterator[ArchiveRecord]:
    """Yield records from a JSON-Lines file (one JSON object per non-blank line)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ContractError(f"line {lineno}: each line must be a JSON object")
            yield _coerce_record(obj, lineno)


def load_archive(path: str | Path) -> Archive:
    """Load and assemble an archive from a JSON-Lines file."""
    return build_archive(iter_jsonl(path))
