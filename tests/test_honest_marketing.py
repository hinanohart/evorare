"""Mirror of the CI honest-marketing gate: banned claims absent, required disclaimers present.

The denylist targets claim forms (e.g. 'robust population diversity', 'measures semantic
diversity', bare 'TMRCA'), not innocuous English like 'the first numeric metric'. README and
src are scanned; the test files themselves are not (they necessarily quote the banned terms).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = [ROOT / "README.md", *(ROOT / "src").rglob("*.py")]

BANNED = re.compile(
    r"state.of.the.art|\bSOTA\b|outperform|guaranteed|robust population diversity|"
    r"measures semantic diversity|\bTMRCA\b|permanent|初の|永続|"
    r"\+[0-9]+(\.[0-9]+)?\s*%",
)

REQUIRED_README = [
    "realized-sample diversity only; does not estimate population diversity",
    "syntactic/structural diversity, not semantic diversity",
    "synthetic",
]
REQUIRED_PRIOR_ART = ["Chao", "rarefaction", "FunSearch"]


def test_no_banned_claims_in_source_or_readme():
    hits = []
    for path in SCANNED:
        text = path.read_text(encoding="utf-8")
        for m in BANNED.finditer(text):
            hits.append(f"{path.name}: {m.group(0)!r}")
    assert not hits, f"banned marketing terms found: {hits}"


def test_required_nonclaims_present():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for literal in REQUIRED_README:
        assert literal in readme, f"missing NON-CLAIM literal: {literal!r}"


def test_required_prior_art_present():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for token in REQUIRED_PRIOR_ART:
        assert token in readme, f"missing prior-art mention: {token!r}"
