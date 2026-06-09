"""Archive record contract and the parsed-archive container.

The primary input is generic JSON-Lines. Required fields: ``id``, ``code``, ``score``.
Optional fields: ``generation``, ``parent_id``, ``island_id``. Missing optional fields
trigger documented, fabrication-free degrade behaviour (see :mod:`evorare.ingest`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchiveRecord:
    """One program in an evolutionary-search archive.

    Attributes:
        id: Unique program identifier (string).
        code: Program source text.
        score: Fitness/quality score. May be ``None`` only after a score-missing degrade,
            in which case the behaviour featurizer is disabled.
        generation: Sampling unit. ``None`` if the archive does not record generations
            (an insertion-order proxy is then used and trend claims are suppressed).
        parent_id: Parent program id, or ``None`` for founders / archives without genealogy.
        island_id: Island/sub-population id, or ``None``.
    """

    id: str
    code: str
    score: float | None
    generation: int | None = None
    parent_id: str | None = None
    island_id: int | None = None


@dataclass(frozen=True)
class Archive:
    """A parsed archive plus machine-readable availability flags.

    ``generation_source`` is ``"explicit"`` when generations were present in the input,
    or ``"insertion-order(proxy)"`` when synthesised from record order. Under the proxy,
    trend/slope claims are forbidden (CHANGE#6) and only cumulative spectra are reported.
    """

    records: tuple[ArchiveRecord, ...]
    has_score: bool
    has_generation: bool
    has_parent_id: bool
    generation_source: str

    def __len__(self) -> int:
        return len(self.records)
