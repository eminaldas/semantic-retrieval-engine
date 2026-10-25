import pytest

from backend.knowledge.service import KnowledgeService

TWO_PROJECTS = {"a": {"root": "/a"}, "b": {"root": "/b"}}


def _service(config):
    return KnowledgeService(runtime=None, config_loader=lambda: config)


def test_default_project_wins_when_none_given():
    assert _service({**TWO_PROJECTS, "_default_project": "b"}).resolve_project(None) == "b"


def test_explicit_project_overrides_default():
    assert _service({**TWO_PROJECTS, "_default_project": "b"}).resolve_project("a") == "a"


def test_without_default_searches_all_projects():
    assert _service(TWO_PROJECTS).resolve_project(None) is None


def test_single_project_is_implicit():
    assert _service({"a": {"root": "/a"}}).resolve_project(None) == "a"


def test_unknown_project_is_rejected():
    with pytest.raises(ValueError):
        _service(TWO_PROJECTS).resolve_project("c")


def test_ask_formats_answer_with_sources():
    from backend.knowledge.agent import AgentResult

    seen = {}

    class FakeAgent:
        def __init__(self, service, project=None):
            seen["project"] = project

        def ask(self, question):
            return AgentResult("Esik 0,91 [KARARLAR_OZET.md].", steps=[("search", {})], seconds=3.2)

    service = _service({"_default_project": "p", "p": {"root": "/x"}})
    service.agent_factory = FakeAgent
    text = service.ask("e5 esigi kac?", project="p")
    assert text.startswith("Esik 0,91") and "Kaynaklar: KARARLAR_OZET.md" in text and "1 arama/okuma" in text
    assert seen["project"] == "p"


def test_ask_rejects_empty_question():
    with pytest.raises(ValueError):
        _service({"p": {"root": "/x"}}).ask("  ")
