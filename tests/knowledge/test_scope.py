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
    assert "search" in names and not names & {"add_note", "add_document", "sync_project"}
    result = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "add_note", "arguments": {"title": "t", "body": "b"}}})["result"]
    assert result["isError"]
    full = MCPServer(_service())
    assert len(full.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]) == 10
