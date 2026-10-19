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
