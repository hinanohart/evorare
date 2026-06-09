import json

import pytest

from evorare.cli import main
from evorare.synth import generate


def _write_archive(tmp_path):
    recs, _ = generate("S-HEALTHY", 0)
    p = tmp_path / "a.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": r.id,
                    "code": r.code,
                    "score": r.score,
                    "generation": r.generation,
                    "parent_id": r.parent_id,
                }
            )
            for r in recs
        ),
        encoding="utf-8",
    )
    return p


def test_cli_diagnose_writes_outputs(tmp_path):
    p = _write_archive(tmp_path)
    out = tmp_path / "r.json"
    svg = tmp_path / "r.svg"
    rc = main(
        ["diagnose", str(p), "--out", str(out), "--svg", str(svg), "--seed", "0", "--n-boot", "60"]
    )
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "HEALTHY"
    assert svg.read_text(encoding="utf-8").startswith("<svg")


def test_cli_diagnose_json_stdout(tmp_path, capsys):
    p = _write_archive(tmp_path)
    rc = main(["diagnose", str(p), "--json", "--n-boot", "40"])
    assert rc == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["verdict"] == "HEALTHY"


def test_cli_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_cli_gate_runs():
    rc = main(["gate", "--n-seed", "3"])
    assert rc == 0  # all gates pass


def test_cli_convert_openevolve(tmp_path):
    d = tmp_path / "ckpt" / "programs"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"p{i}.json").write_text(
            json.dumps(
                {
                    "id": f"p{i}",
                    "code": "def f():\n    return 1\n",
                    "generation": i,
                    "parent_id": None if i == 0 else f"p{i - 1}",
                    "metrics": {"combined_score": 0.1 * i},
                }
            ),
            encoding="utf-8",
        )
    out = tmp_path / "o.jsonl"
    rc = main(["convert", "openevolve", str(tmp_path / "ckpt"), "-o", str(out)])
    assert rc == 0
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 3
