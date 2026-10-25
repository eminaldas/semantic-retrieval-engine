"""Faz 4: tarih/guncellik etiketi, filtreler, toplu sorgu, supersedes, add_document, izleyici."""

import os
from datetime import datetime, timezone

import pytest
from qdrant_client import models

from backend.knowledge import projects, search, service as service_mod, store, watcher
from backend.knowledge.service import KnowledgeService


class TestPayloadHelpers:
    def test_dir_prefixes(self):
        assert projects.dir_prefixes("graf/alt/x.md") == ["graf/", "graf/alt/"]
        assert projects.dir_prefixes("KOK.md") == []

    def test_superseded_marker_round_trip(self):
        text = "# Eski\n\nicerik\n" + projects.SUPERSEDED_MARKER.format(path="notes/2026-yeni.md") + "\n"
        assert projects.superseded_by(text) == "notes/2026-yeni.md"
        assert projects.superseded_by("# baska\n_Proje: x_\n") is None

    def test_doc_type(self):
        assert projects.doc_type_for("notes/2026-x.md") == "note"
        assert projects.doc_type_for("notes/belgeler/plan.md") == "doc"
        assert projects.doc_type_for("graf/x.md") == "doc"

    def test_document_relpath_cannot_escape(self):
        assert projects.document_relpath("../../etc/Plan Özeti.md") == "notes/belgeler/etc-plan-ozeti.md"

    def test_records_carry_new_fields(self):
        [rec] = projects.build_chunk_records("p", "graf/x.md", "# A\nmetin", [("metin", "satır 2-2")], mtime=0,
                                             indexed_at="2026-10-22T00:00:00+00:00")
        assert rec["dirs"] == ["graf/"] and rec["indexed_at"].startswith("2026-10-22") and rec["superseded_by"] is None


class TestFilter:
    def _conditions(self, **kwargs):
        return {c.key: c for c in store.project_filter(**kwargs).must}

    def test_folder_prefix_uses_dirs(self):
        cond = self._conditions(path_prefix="./graf")["dirs"]
        assert cond.match == models.MatchValue(value="graf/")

    def test_file_prefix_uses_exact_path(self):
        assert self._conditions(path_prefix="/KARARLAR_OZET.md")["path"].match.value == "KARARLAR_OZET.md"

    def test_date_range_inclusive(self):
        rng = self._conditions(modified_after="2026-10-01", modified_before="2026-10-22")["modified_at"].range
        assert rng.gte.date().isoformat() == "2026-10-01" and rng.lte.isoformat().startswith("2026-10-22T23:59:59")

    def test_empty_filter(self):
        assert store.project_filter() is None


def _result(path="a.md", score=1.0, low=False, **extra):
    return dict({"path": path, "location": "satır 1-2", "section": "", "text": "t", "rerank_score": score,
                 "low_confidence": low, "modified_at": "2026-10-20T10:00:00+00:00"}, **extra)


class TestFlags:
    def test_superseded_note_is_demoted_below_fresh_one(self):
        ranked = search.demote_superseded([_result("old", 0.9, superseded_by="notes/new.md"), _result("new", 0.6)])
        assert [r["path"] for r in ranked] == ["new", "old"]

    def test_header_shows_date_and_flags(self):
        out = search.format_results([_result(superseded_by="notes/n.md", stale="dosya silinmis")])
        first = out.splitlines()[0]
        assert "2026-10-20" in first and "eski not, yerini notes/n.md aldi" in first and "guncel degil: dosya silinmis" in first


class FakeRuntime:
    def client(self):
        return None

    def embedder(self):
        return None

    def reranker(self):
        return None

    def chunker_model(self):
        return None

    def contextualizer(self):
        return None


def _service(tmp_path, monkeypatch, results=None):
    root = tmp_path / "repo"
    root.mkdir()
    config = {"p": {"root": str(root)}}
    svc = KnowledgeService(FakeRuntime(), config_loader=lambda: config, notes_dir=str(tmp_path / "notes"))
    synced = []
    monkeypatch.setattr(service_mod.sync, "sync_project",
                        lambda project, *a, only=None, **k: synced.append((project, only)) or
                        {"indexed": list(only or []), "removed": [], "chunks": 1})
    monkeypatch.setattr(service_mod.search, "search", lambda *a, **k: [dict(r) for r in (results or [])])
    return svc, root, synced


class TestService:
    def test_stale_when_file_changed_after_index(self, tmp_path, monkeypatch):
        svc, root, _ = _service(tmp_path, monkeypatch)
        (root / "a.md").write_text("x")
        old = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        fresh = datetime.fromtimestamp(os.path.getmtime(root / "a.md"), timezone.utc).isoformat()
        stale, ok, gone = svc.mark_stale([
            _result("a.md", project="p", modified_at=old),
            _result("a.md", project="p", modified_at=fresh),
            _result("yok.md", project="p", modified_at=old),
        ])
        assert "indeks eski" in stale["stale"] and "stale" not in ok and gone["stale"] == "dosya silinmis"

    def test_batch_search_has_one_section_per_question(self, tmp_path, monkeypatch):
        svc, _, _ = _service(tmp_path, monkeypatch, results=[_result(project="p")])
        out = svc.search(queries=["bir", "iki"])
        assert "## Soru 1: bir" in out and "## Soru 2: iki" in out

    def test_batch_limit_and_bad_date(self, tmp_path, monkeypatch):
        svc, _, _ = _service(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            svc.search(queries=["a", "b", "c", "d", "e"])
        with pytest.raises(ValueError):
            svc.search("a", modified_after="22.10.2026")

    def test_supersedes_marks_old_note_and_syncs_both(self, tmp_path, monkeypatch):
        svc, _, synced = _service(tmp_path, monkeypatch)
        notes = tmp_path / "notes" / "p"
        notes.mkdir(parents=True)
        (notes / "eski.md").write_text("# Eski\n", encoding="utf-8")
        msg = svc.add_note("Yeni esik", "0.05", project="p", supersedes="notes/eski.md")
        new_rel = msg.split(": ")[1].split(".md")[0] + ".md"
        assert projects.superseded_by((notes / "eski.md").read_text(encoding="utf-8")) == new_rel
        assert synced == [("p", [new_rel, "notes/eski.md"])]

    def test_supersedes_rejects_non_note_path(self, tmp_path, monkeypatch):
        svc, _, _ = _service(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            svc.add_note("t", "b", project="p", supersedes="../KARARLAR.md")

    def test_add_document_refuses_overwrite_unless_asked(self, tmp_path, monkeypatch):
        svc, _, synced = _service(tmp_path, monkeypatch)
        assert "notes/belgeler/faz-4-plani.md" in svc.add_document("Faz 4 planı", "# Plan", project="p")
        with pytest.raises(ValueError):
            svc.add_document("Faz 4 planı", "# Yeni", project="p")
        svc.add_document("Faz 4 planı", "# Yeni", project="p", overwrite=True)
        assert (tmp_path / "notes" / "p" / "belgeler" / "faz-4-plani.md").read_text(encoding="utf-8") == "# Yeni\n"
        assert len(synced) == 2


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestWatcher:
    def test_change_is_synced_after_it_settles(self, tmp_path, monkeypatch):
        svc, root, synced = _service(tmp_path, monkeypatch)
        (root / "a.md").write_text("bir")
        clock = FakeClock()
        w = watcher.ProjectWatcher(svc, settle=3, log=lambda m: None, clock=clock)
        w.catch_up()
        assert synced == [("p", None)]  # acilis senkronu
        (root / "a.md").write_text("iki satir")
        clock.now = 1
        w.poll_once()
        assert len(synced) == 1  # henuz yerlesmedi
        clock.now = 5
        w.poll_once()
        assert synced[-1] == ("p", None) and len(synced) == 2
        clock.now = 10
        w.poll_once()
        assert len(synced) == 2  # degisiklik yoksa senkron yok

    def test_periodic_reconcile_without_visible_change(self, tmp_path, monkeypatch):
        svc, _, synced = _service(tmp_path, monkeypatch)
        clock = FakeClock()
        w = watcher.ProjectWatcher(svc, reconcile=120, log=lambda m: None, clock=clock)
        w.catch_up()
        clock.now = 60
        w.poll_once()
        assert len(synced) == 1
        clock.now = 125
        w.poll_once()
        assert len(synced) == 2

    def test_watch_can_be_disabled(self, tmp_path, monkeypatch):
        svc, _, synced = _service(tmp_path, monkeypatch)
        svc.config_loader = lambda: {"_watch": False, "p": {"root": str(tmp_path / "repo")}}
        watcher.ProjectWatcher(svc, log=lambda m: None).catch_up()
        assert synced == []
