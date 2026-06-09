import json

import pytest

from evorare.ingest import ContractError, build_archive, load_archive
from evorare.schema import ArchiveRecord


def test_build_archive_explicit_generation():
    recs = [ArchiveRecord(id=str(i), code="x=1", score=0.1, generation=i % 2) for i in range(4)]
    arc = build_archive(recs)
    assert arc.has_generation
    assert arc.generation_source == "explicit"


def test_build_archive_proxy_generation():
    recs = [ArchiveRecord(id=str(i), code="x=1", score=0.1) for i in range(4)]
    arc = build_archive(recs)
    assert not arc.has_generation
    assert arc.generation_source == "insertion-order(proxy)"
    assert [r.generation for r in arc.records] == [0, 1, 2, 3]


def test_build_archive_no_score_flag():
    recs = [ArchiveRecord(id=str(i), code="x=1", score=None, generation=0) for i in range(3)]
    arc = build_archive(recs)
    assert not arc.has_score


def test_build_archive_parent_flag():
    recs = [
        ArchiveRecord(id="a", code="x", score=0.1, generation=0),
        ArchiveRecord(id="b", code="x", score=0.1, generation=1, parent_id="a"),
    ]
    assert build_archive(recs).has_parent_id


def test_load_jsonl(tmp_path):
    p = tmp_path / "arc.jsonl"
    lines = [
        {"id": "a", "code": "x=1", "score": 0.5, "generation": 0},
        {"id": "b", "code": "x=2", "score": 0.6, "generation": 1, "parent_id": "a"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    arc = load_archive(p)
    assert len(arc) == 2
    assert arc.has_parent_id


def test_load_jsonl_blank_lines(tmp_path):
    p = tmp_path / "arc.jsonl"
    p.write_text('{"id":"a","code":"x","score":0.1}\n\n', encoding="utf-8")
    assert len(load_archive(p)) == 1


def test_contract_requires_id_code(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"a","score":0.1}\n', encoding="utf-8")
    with pytest.raises(ContractError):
        load_archive(p)


def test_contract_rejects_non_object(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("[1,2,3]\n", encoding="utf-8")
    with pytest.raises(ContractError):
        load_archive(p)


def test_score_bool_rejected(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"a","code":"x","score":true}\n', encoding="utf-8")
    with pytest.raises(ContractError):
        load_archive(p)
