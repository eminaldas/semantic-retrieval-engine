"""Turlu kayitlar (plan H2): kisa kimlik, tur, acik is, checkpoint, close_task."""

import pytest

from backend.knowledge import projects
from backend.knowledge.scope import Scope
from backend.knowledge.service import KnowledgeService

CONFIG = {"_default_project": "a", "a": {"root": "/a"}}


@pytest.fixture
def service(tmp_path, monkeypatch):
    svc = KnowledgeService(runtime=None, config_loader=lambda: CONFIG, notes_dir=str(tmp_path))
    svc.synced = []
    monkeypatch.setattr(svc, "_sync", lambda project, full=False, only=None: svc.synced.append(tuple(only or ()))
                        or {"indexed": [], "removed": [], "chunks": 1})
    return svc


def _meta(service, rel):
    return projects.note_meta(service._read_note("a", rel))


def test_records_get_a_kind_and_a_short_id(service):
    first = service.add_note("Esik 0,05", "karar", kind="decision")
    assert "K-0001" in first and "decision" in first
    rel = service.synced[-1][0]
    assert _meta(service, rel) == {"record_id": "K-0001", "kind": "decision", "status": None}
    assert "I-0002" in service.add_note("Gold seti genislet", "is", kind="task")  # numara proje icinde artar
    with pytest.raises(ValueError, match="kind su degerlerden"):
        service.add_note("x", "y", kind="karar")


def test_superseding_accepts_a_short_id(service):
    service.add_note("Esik 0,05", "eski", kind="decision")
    old = service.synced[-1][0]
    message = service.add_note("Esik 0,07", "yeni", kind="decision", supersedes="K-0001")
    assert f"Eski kayit ({old})" in message
    assert projects.superseded_by(service._read_note("a", old)) == service.synced[-1][0]
    assert sorted(service.synced[-1]) == sorted([old, service.synced[-1][0]])
    with pytest.raises(ValueError, match="K-0099 kimlikli kayit"):
        service.add_note("x", "y", supersedes="K-0099")


def test_checkpoint_writes_the_session_decisions_and_tasks(service):
    message = service.checkpoint("Faz 4 bitti", decisions=["Esik 0,05 kalsin"], open_tasks=["Arayuzu yaz"])
    assert "1 karar, 1 acik is" in message and "K-0001" in message and "I-0002" in message
    written = service.synced[-1]
    assert len(written) == 3  # karar, is, oturum ozeti tek senkronda
    session = next(r for r in written if _meta(service, r)["kind"] == "session")
    body = service._read_note("a", session)
    assert "Faz 4 bitti" in body and "Esik 0,05 kalsin" in body and "Arayuzu yaz" in body
    assert _meta(service, session)["record_id"] == "O-0003"
    with pytest.raises(ValueError, match="summary bos"):
        service.checkpoint("  ")


def test_close_task_marks_it_done_and_refuses_other_kinds(service):
    service.checkpoint("ozet", open_tasks=["Arayuzu yaz"])
    task = next(r for r in service.synced[-1] if _meta(service, r)["kind"] == projects.TASK_KIND)
    assert _meta(service, task)["status"] == "open"
    assert "Is kapatildi" in service.close_task("I-0001", note="PR #13")
    assert _meta(service, task)["status"] == "done" and "PR #13" in service._read_note("a", task)
    assert "zaten kapali" in service.close_task("I-0001")
    service.add_note("Karar", "govde", kind="decision")
    with pytest.raises(ValueError, match="bir is kaydi degil"):
        service.close_task("K-0003")
    with pytest.raises(ValueError, match="Kisa kimlik"):
        service.close_task("../../etc/passwd")


def test_writes_stay_inside_the_scope(service):
    view = service.scoped(Scope.workspace("w", read=["a"]))
    with pytest.raises(ValueError, match="salt okunur"):
        view.checkpoint("ozet")
    with pytest.raises(ValueError, match="salt okunur"):
        view.close_task("I-0001")
