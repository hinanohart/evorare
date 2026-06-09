#!/usr/bin/env python3
"""Run the sensitivity gates at the canonical config and write the metrics file.

Writes results/v0.1.0a2_metrics.json with an env stamp, the per-gate pass booleans, parsed
headline numbers, and the test count. README numbers are generated from this file (S5); no
hand-written numbers exist before this runs.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evorare.gate import GATE_N_BOOT, run_all_gates  # noqa: E402
from evorare.report.json_report import env_stamp  # noqa: E402

SEED = 0
N_SEED = 20


def _f(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _count_tests() -> int:
    n = 0
    for tf in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(tf.read_text(encoding="utf-8"))
        n += sum(
            1
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
    return n


def main() -> int:
    report = run_all_gates(seed=SEED, n_seed=N_SEED)
    gates = {k: bool(r.passed) for k, r in report.results.items()}
    details = {k: r.detail for k, r in report.results.items()}

    headline = {
        "g2_auc_healthy_vs_saturate": _f(r"AUC=([0-9.]+)", details["G2"]),
        "g3_bottleneck_detect_within2": _f(r"detect_within2=([0-9.]+)", details["G3"]),
        "g3_collapse_rate": _f(r"collapse_rate=([0-9.]+)", details["G3"]),
        "g3_stationary_false_collapse": _f(r"stationary_fpr=([0-9.]+)", details["G3"]),
        "g4_not_saturating_rate": _f(r"not_saturating_rate=([0-9.]+)", details["G4"]),
        "g5_resolution_disagreement": _f(r"disagreement_rate=([0-9.]+)", details["G5"]),
        "g6_exploit_indeterminate": _f(r"indeterminate_rate=([0-9.]+)", details["G6"]),
        "g8_aggregated_coverage_excluded": _f(r"excluded_rate=([0-9.]+)", details["G8"]),
        "g8_null_false_flag": _f(r"false_flag_rate=([0-9.]+)", details["G8"]),
    }

    metrics = {
        "env": env_stamp(SEED),
        "config": {
            "seed": SEED,
            "n_seed": N_SEED,
            "n_boot": GATE_N_BOOT,
            "n_gen": 8,
            "n_per_gen": 36,
            "scenarios": [
                "S-NULL",
                "S-HEALTHY",
                "S-SATURATE",
                "S-STATIONARY",
                "S-EXPLOIT",
                "S-AGGREGATED",
                "S-BOTTLENECK",
            ],
        },
        "gates": gates,
        "gate_details": details,
        "headline": headline,
        "n_tests": _count_tests(),
        "all_gates_passed": report.all_passed,
        "honest_synthetic_done": True,
        "real_archive_smoke": False,
    }

    out = ROOT / "results" / "v0.1.0a2_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    print(f"all_gates_passed={report.all_passed}  n_tests={metrics['n_tests']}")
    for k in sorted(gates):
        print(f"  {k}: {'PASS' if gates[k] else 'FAIL'}  {details[k]}")
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
