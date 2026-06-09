from evorare.schema import Archive, ArchiveRecord


def test_record_fields():
    r = ArchiveRecord(id="a", code="x=1", score=0.5)
    assert r.id == "a"
    assert r.score == 0.5
    assert r.generation is None
    assert r.parent_id is None


def test_record_is_frozen():
    r = ArchiveRecord(id="a", code="x=1", score=0.5)
    try:
        r.id = "b"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("ArchiveRecord should be frozen")


def test_archive_len():
    recs = (ArchiveRecord(id=str(i), code="x", score=0.1, generation=0) for i in range(3))
    arc = Archive(tuple(recs), True, True, False, "explicit")
    assert len(arc) == 3
