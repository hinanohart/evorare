"""R17 step-5 gate: every sensitivity gate G1..G9 must pass on synthetic ground truth."""

from evorare.gate import run_all_gates


def test_all_gates_pass():
    report = run_all_gates(seed=0, n_seed=8)
    failed = {k: r.detail for k, r in report.results.items() if not r.passed}
    assert not failed, f"gates failed: {failed}"
    assert report.all_passed


def test_gate_count_is_nine():
    report = run_all_gates(seed=0, n_seed=3)
    assert len(report.results) == 9
    assert {f"G{i}" for i in range(1, 10)} == set(report.results)
