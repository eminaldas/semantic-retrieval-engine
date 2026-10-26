import pytest

from backend.knowledge import routing, search, store
from backend.knowledge.scope import Scope
from backend.knowledge.service import KnowledgeService
from backend.mcp.server import MCPServer

CONFIG = {"_default_project": "a", "a": {"root": "/a", "description": "A projesi"}, "b": {"root": "/b"},
          "c": {"root": "/c"}}


def _service(scope=None, notes_dir=None):
    service = KnowledgeService(runtime=None, config_loader=lambda: CONFIG, notes_dir=notes_dir or "/yok")
    return service.scoped(scope) if scope else service


def test_workspace_scope_rules():
    scope = Scope.workspace("w", read=["b"], write=["c"], default="c")
    assert scope.can_read("b") and scope.can_read("c") and not scope.can_read("a")
    assert scope.can_write("c") and not scope.can_write("b") and not scope.can_write("a")
    assert Scope.admin().can_write("x") and Scope.workspace("r", read=["b"]).read_only
    with pytest.raises(ValueError):
        Scope.workspace("w", read=["b"], default="a")


def test_out_of_scope_project_is_rejected_without_revealing_it():
    service = _service(Scope.workspace("w", read=["b", "c"]))
    with pytest.raises(ValueError) as exc:
        service.resolve_project("a")
    assert "'a'" in str(exc.value) and "b, c" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        service.resolve_project("yok")
    assert "a" not in str(exc.value).split("Erisilebilir projeler:")[1]


def test_default_comes_from_scope_then_config():
    assert _service(Scope.workspace("w", read=["b", "c"], default="c")).resolve_project(None) == "c"
    assert _service(Scope.workspace("w", read=["a", "b"])).resolve_project(None) == "a"  # config varsayilani kapsamda
    assert _service(Scope.workspace("w", read=["b"])).resolve_project(None) == "b"  # tek proje


def test_unscoped_query_is_limited_to_scope_projects():
    service = _service(Scope.workspace("w", read=["b", "c"]))
    assert service.resolve_project(None) is None
    assert service._project_filter(None) == ("b", "c")
    assert _service()._project_filter(None) is None  # yonetici: bugunku davranis


def test_list_projects_hides_other_projects_and_paths():
    text = _service(Scope.workspace("w", read=["a"], write=["c"])).list_projects()
    assert "- a: A projesi [salt okunur]" in text and "- c" in text
    assert "/a" not in text and "- b" not in text
    assert "(/b)" in _service().list_projects()


def test_writes_need_a_writable_project(tmp_path, monkeypatch):
    service = _service(Scope.workspace("w", read=["a"], write=["c"], default="c"), notes_dir=str(tmp_path))
    synced = []
    monkeypatch.setattr(service, "_sync", lambda project, full=False, only=None: synced.append(project)
                        or {"indexed": [], "removed": [], "chunks": 1})
    with pytest.raises(ValueError, match="salt okunur"):
        service.add_note("baslik", "govde", project="a")
    with pytest.raises(ValueError, match="salt okunur"):
        service.add_document("plan", "icerik", project="a")
    with pytest.raises(ValueError, match="salt okunur"):
        service.sync_project("a")
    with pytest.raises(ValueError, match="Bu workspace'te"):
        service.add_note("baslik", "govde", project="b")
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    assert service.add_note("baslik", "govde").startswith("Not kaydedildi")  # ev projesi: c
    assert synced == ["c"] and (tmp_path / "c").is_dir()


def test_search_without_project_uses_scope_filter_and_skips_hint(monkeypatch):
    service = _service(Scope.workspace("w", read=["b", "c"]))
    seen = {}
    monkeypatch.setattr(service, "_search_one", lambda q, k, o: seen.setdefault("project", o["project"]) and [])
    monkeypatch.setattr(search, "format_results", lambda results, max_chars=None: "SONUC")
    monkeypatch.setattr(routing, "project_hint", lambda *a, **k: pytest.fail("kapsam aramasinda ipucu olmamali"))
    assert service.search("soru") == "SONUC" and seen["project"] == ("b", "c")


def test_hint_only_considers_scope_projects(monkeypatch):
    service = _service(Scope.workspace("w", read=["a", "b"]))
    seen = {}
    monkeypatch.setattr(service, "_search_one", lambda q, k, o: [])
    monkeypatch.setattr(search, "format_results", lambda results, max_chars=None: "SONUC")

    class Runtime:
        def client(self): return None
        def embedder(self): return None
        def reranker(self): return None

    service.runtime = Runtime()

    def fake_hint(client, embedder, reranker, query, current, projects, current_score=None):
        seen["projects"] = projects

    monkeypatch.setattr(routing, "project_hint", fake_hint)
    service.search("soru")
    assert seen["projects"] == ["a", "b"]


def test_project_filter_matches_any_scope_project():
    flt = store.project_filter(project=("b", "c"), doc_type="doc")
    keys = {c.key: c.match for c in flt.must}
    assert keys["project"].any == ["b", "c"] and keys["doc_type"].value == "doc"


def test_scoped_view_shares_state_and_ask_inherits_scope():
    admin = _service()
    view = admin.scoped(Scope.workspace("w", read=["b"]))
    assert view.sync_lock is admin.sync_lock and view.last_sync is admin.last_sync
    assert admin.scope.is_admin

    seen = {}

    class FakeAgent:
        def __init__(self, service, project=None):
            seen["scope"], seen["project"] = service.scope, service.resolve_project(project)

        def ask(self, question):
            from backend.knowledge.agent import AgentResult
            return AgentResult("cevap", steps=[], seconds=1.0)

    view.agent_factory = FakeAgent
    view.ask("soru")
    assert seen["scope"].name == "w" and seen["project"] == "b"


def test_read_only_workspace_hides_write_tools():
    server = MCPServer(_service(Scope.workspace("r", read=["b"])))
    names = {t["name"] for t in server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert "search" in names and not names & {"add_note", "add_document", "sync_project", "checkpoint", "close_task"}
    result = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "add_note", "arguments": {"title": "t", "body": "b"}}})["result"]
    assert result["isError"]
    full = MCPServer(_service())
    assert len(full.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]) == 13


class _Point:
    def __init__(self, payload):
        self.payload = payload


class _Client:
    def __init__(self, rows):
        self.rows, self.filters = rows, []

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.filters.append(scroll_filter)
        return [_Point(r) for r in self.rows], None


def _row(project, path, doc_type, modified, section="", superseded_by=None, kind=None, status=None, record_id=None):
    return {"source_id": f"{project}:{path}", "project": project, "path": path, "doc_type": doc_type,
            "modified_at": modified, "section": section, "superseded_by": superseded_by, "kind": kind,
            "status": status, "record_id": record_id}


def test_brief_lists_live_notes_and_recent_docs_within_scope():
    rows = [_row("b", "notes/n1.md", "note", "2026-10-20T10:00:00Z", "Eski esik"),
            _row("b", "notes/n1.md", "note", "2026-10-20T10:00:00Z", "Eski esik > Ayrinti",
                 superseded_by="notes/n2.md"),
            _row("b", "notes/n2.md", "note", "2026-10-25T10:00:00Z", "Yeni esik 0,05"),
            _row("c", "PLAN.md", "doc", "2026-10-24T10:00:00Z", "Plan")]
    client = _Client(rows)
    service = _service(Scope.workspace("w", read=["b"], write=["c"]))
    service.runtime = type("R", (), {"client": lambda self: client})()
    text = service.brief()
    assert text.startswith("# Brifing · workspace w") and "Proje b [salt okunur]" in text and "Proje c" in text
    assert "2026-10-25 b: notes/n2.md: Yeni esik 0,05" in text and "n1.md" not in text
    assert "2026-10-24 c: PLAN.md" in text and len(text) <= 1500
    assert client.filters[0].must[0].match.any == ["b", "c"]
    with pytest.raises(ValueError):
        service.brief(project="a")


def test_brief_lists_open_tasks_and_hides_closed_ones():
    rows = [_row("b", "notes/t1.md", "note", "2026-10-25T10:00:00Z", "Arayuzu yaz", kind="task", status="open",
                 record_id="I-0007"),
            _row("b", "notes/t2.md", "note", "2026-10-24T10:00:00Z", "Esik ayarla", kind="task", status="done",
                 record_id="I-0004")]
    client = _Client(rows)
    service = _service(Scope.workspace("w", read=["b"]))
    service.runtime = type("R", (), {"client": lambda self: client})()
    text = service.brief()
    assert "Acik isler:" in text and "- I-0007 Arayuzu yaz" in text and "I-0004" not in text


class _PricedAgent:
    def __init__(self, service, project=None):
        pass

    def ask(self, question):
        from backend.knowledge.agent import AgentResult
        return AgentResult("cevap [a.md]", prompt_tokens=200_000, output_tokens=1_000, seconds=1.0)


def test_external_llm_off_hides_and_blocks_ask():
    view = _service(Scope.workspace("w", read=["b"], external_llm=False))
    view.agent_factory = _PricedAgent
    with pytest.raises(ValueError, match="dis LLM kapali"):
        view.ask("soru")
    names = {t["name"] for t in MCPServer(view).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert "ask" not in names and "search" in names


def test_ask_daily_budget_is_per_workspace_and_day():
    admin = _service()
    view = admin.scoped(Scope.workspace("w", read=["b"], ask_daily_usd=0.0001))
    other = admin.scoped(Scope.workspace("v", read=["c"], ask_daily_usd=0.0001))
    for s in (view, other):
        s.agent_factory = _PricedAgent
    admin.today = view.today = other.today = lambda: "2026-10-26"
    assert view.ask("soru").startswith("cevap")
    with pytest.raises(ValueError, match="gunluk ask siniri"):
        view.ask("soru")
    assert other.ask("soru").startswith("cevap")  # baska workspace'in sayaci ayri
    view.today = lambda: "2026-10-27"
    assert view.ask("soru").startswith("cevap")  # yeni gun
