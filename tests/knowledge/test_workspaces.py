import pytest

from backend.knowledge.workspaces import describe, load_workspaces

PROJECTS = {"_default_project": "a", "a": {"root": "/a"}, "b": {"root": "/b"}}


def _load(section):
    return load_workspaces({**PROJECTS, "_workspaces": section})


def test_missing_section_means_no_workspaces():
    assert load_workspaces(PROJECTS) == {}


def test_workspace_becomes_scope_with_single_write_project_as_home():
    ws = _load({"w": {"description": "Ana", "projects": {"a": "write", "b": "read"}, "instructions": "Kural"}})["w"]
    assert ws.scope.can_write("a") and ws.scope.can_read("b") and not ws.scope.can_write("b")
    assert ws.scope.default == "a" and ws.instructions == "Kural"
    assert "a (yazar, ev projesi)" in describe(ws) and "b (okur)" in describe(ws)


def test_read_only_workspace_has_no_home():
    ws = _load({"r": {"projects": {"b": "read"}}})["r"]
    assert ws.scope.read_only and ws.scope.default is None


def test_two_write_projects_need_explicit_home():
    ws = _load({"w": {"projects": {"a": "write", "b": "write"}, "home": "b"}})["w"]
    assert ws.scope.default == "b"
    assert _load({"w": {"projects": {"a": "write", "b": "write"}}})["w"].scope.default is None


@pytest.mark.parametrize("section, message", [
    ({"W x": {"projects": {"a": "read"}}}, "ad kucuk harf"),
    ({"w": {"projects": {}}}, "en az bir proje"),
    ({"w": {"projects": {"c": "read"}}}, "'c' projesi tanimli degil"),
    ({"w": {"projects": {"a": "admin"}}}, "rol 'read' ya da 'write'"),
    ({"w": {"projects": {"a": "read"}, "home": "a"}}, "home 'a'"),
    ({"w": {"projects": {"a": "write"}, "proejcts": {}}}, "bilinmeyen alan 'proejcts'"),
    ({"w": {"projects": {"a": "write"}, "instructions": "x" * 2001}}, "en fazla 2000"),
])
def test_invalid_sections_are_rejected(section, message):
    with pytest.raises(ValueError, match=message):
        _load(section)


def test_all_errors_are_reported_together():
    with pytest.raises(ValueError) as exc:
        _load({"x": {"projects": {"c": "read"}}, "y": {"projects": {"a": "boss"}}})
    assert "x: 'c'" in str(exc.value) and "y: 'a'" in str(exc.value)


def test_external_llm_and_ask_budget_reach_the_scope():
    scope = _load({"w": {"projects": {"a": "write"}, "external_llm": False, "ask_daily_usd": 0.5}})["w"].scope
    assert not scope.external_llm and scope.ask_daily_usd == 0.5
    assert _load({"w": {"projects": {"a": "write"}}})["w"].scope.external_llm  # varsayilan acik


@pytest.mark.parametrize("extra, message", [
    ({"external_llm": "no"}, "'external_llm' true ya da false"),
    ({"ask_daily_usd": -1}, "'ask_daily_usd' sifir ya da pozitif"),
    ({"ask_daily_usd": True}, "'ask_daily_usd' sifir ya da pozitif"),
])
def test_invalid_policy_fields_are_rejected(extra, message):
    with pytest.raises(ValueError, match=message):
        _load({"w": {"projects": {"a": "write"}, **extra}})
