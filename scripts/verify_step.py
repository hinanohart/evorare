#!/usr/bin/env python3
"""Per-phase verification predicates (machine gate). Exit 0 = pass, 1 = fail.

Usage: python scripts/verify_step.py S1
Each Sn runs concrete checks (file existence, importability, callable surface).
This is intentionally non-vacuous: it imports and exercises real code, not fixed True.
Imports are written as literal statements (no dynamic importlib) so the module set is
a closed, statically-auditable list.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _exists(*rel: str) -> bool:
    ok = True
    for r in rel:
        p = ROOT / r
        if not p.exists():
            print(f"  missing: {r}")
            ok = False
    return ok


def s1() -> bool:
    ok = _exists(
        "pyproject.toml",
        "LICENSE",
        "README.md",
        ".gitignore",
        "src/evorare/__init__.py",
        "scripts/verify_step.py",
        "scripts/heartbeat.py",
        "src/evorare/synth/generators.py",
    )
    sys.path.insert(0, str(SRC))
    try:
        import evorare  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"  import evorare FAILED: {e}")
        ok = False
    finally:
        sys.path.pop(0)
    return ok


def s2() -> bool:
    sys.path.insert(0, str(SRC))
    try:
        import evorare.cli  # noqa: F401
        import evorare.diversity  # noqa: F401
        import evorare.featurize  # noqa: F401
        import evorare.genealogy  # noqa: F401
        import evorare.ingest  # noqa: F401
        import evorare.judge  # noqa: F401
        import evorare.report  # noqa: F401
        import evorare.sampling_validity  # noqa: F401
        import evorare.schema  # noqa: F401

        return True
    except Exception as e:  # noqa: BLE001
        print(f"  S2 import FAILED: {e}")
        return False
    finally:
        sys.path.pop(0)


def s3() -> bool:
    sys.path.insert(0, str(SRC))
    try:
        import evorare.gate  # noqa: F401
        import evorare.synth.generators  # noqa: F401

        return True
    except Exception as e:  # noqa: BLE001
        print(f"  S3 import FAILED: {e}")
        return False
    finally:
        sys.path.pop(0)


def _pytest() -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-1500:])
    return r.returncode == 0


def s4() -> bool:
    mpath = ROOT / "results" / "v0.1.0a1_metrics.json"
    if not mpath.exists():
        print("  results/v0.1.0a1_metrics.json missing")
        return False
    m = json.loads(mpath.read_text(encoding="utf-8"))
    gates = m.get("gates", {})
    required = [f"G{i}" for i in range(1, 10)]
    missing = [g for g in required if g not in gates]
    if missing:
        print(f"  metrics missing gates: {missing}")
        return False
    failed = [g for g in required if not gates[g]]
    if failed:
        print(f"  gates not passing: {failed}")
        return False
    return _pytest()


STEPS = {"S1": s1, "S2": s2, "S3": s3, "S4": s4}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(f"usage: verify_step.py {{{'|'.join(STEPS)}}}")
        return 2
    step = sys.argv[1]
    ok = STEPS[step]()
    print(f"{step}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
