"""Command-line interface: ``evorare diagnose | gate | convert``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from .featurize import DEFAULT_FEATURIZERS
    from .ingest import load_archive
    from .judge import diagnose
    from .report import diagnosis_to_dict, write_json, write_svg

    featurizers = (
        tuple(f.strip() for f in args.featurizer.split(",") if f.strip())
        if args.featurizer
        else DEFAULT_FEATURIZERS
    )
    archive = load_archive(args.archive)
    diag = diagnose(archive, featurizers=featurizers, seed=args.seed, n_boot=args.n_boot)
    payload = diagnosis_to_dict(diag, archive)

    if args.out:
        write_json(diag, archive, args.out)
    if args.svg:
        write_svg(diag, args.svg)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"verdict: {diag.verdict}")
        print(f"trend_available: {diag.trend_available} ({diag.generation_source})")
        for res in diag.resolutions:
            lo, hi = res.hill_q1_slope_ci
            print(
                f"  [{res.featurizer}] {res.verdict}  "
                f"hill_q1_slope={res.hill_q1_slope:+.4f} "
                f"CI=[{lo:.4f},{hi:.4f}]  coverage_valid={res.coverage_valid_fraction:.2f}"
            )
        if diag.genealogy.available:
            print(
                f"  genealogy: collapse={diag.genealogy_collapse} "
                f"bottleneck_gen={diag.bottleneck_generation}"
            )
        for note in diag.notes:
            print(f"  note: {note}")
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .gate import run_all_gates

    report = run_all_gates(seed=args.seed, n_seed=args.n_seed)
    for name in sorted(report.results):
        r = report.results[name]
        print(f"{name}: {'PASS' if r.passed else 'FAIL'}  {r.detail}")
    print(f"\nALL: {'PASS' if report.all_passed else 'FAIL'}")
    return 0 if report.all_passed else 1


def _cmd_convert(args: argparse.Namespace) -> int:
    from .adapters import convert_to_jsonl

    n = convert_to_jsonl(args.framework, args.path, args.out)
    print(f"wrote {n} records to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evorare",
        description=(
            "Realized-sample diversity-trend diagnostic for LLM evolutionary-search archives. "
            "Does not estimate population diversity."
        ),
    )
    parser.add_argument("--version", action="version", version=f"evorare {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("diagnose", help="diagnose an archive (JSON-Lines)")
    d.add_argument("archive")
    d.add_argument("--featurizer", default=None, help="comma list, e.g. behavior,ast")
    d.add_argument("--out", default=None, help="write JSON result here")
    d.add_argument("--svg", default=None, help="write SVG trend here")
    d.add_argument("--json", action="store_true", help="print full JSON to stdout")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--n-boot", dest="n_boot", type=int, default=200)
    d.set_defaults(func=_cmd_diagnose)

    g = sub.add_parser("gate", help="run sensitivity gates G1..G9 (exit 0/1)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n-seed", dest="n_seed", type=int, default=20)
    g.set_defaults(func=_cmd_gate)

    c = sub.add_parser("convert", help="convert a framework archive to JSON-Lines")
    c.add_argument("framework", choices=["openevolve", "shinka"])
    c.add_argument("path")
    c.add_argument("-o", "--out", required=True)
    c.set_defaults(func=_cmd_convert)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result = func(args)
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
