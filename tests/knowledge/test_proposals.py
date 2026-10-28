import json
import os

import pytest

from backend.knowledge import proposals as pr
from backend.knowledge.scope import Scope
from backend.knowledge.workspaces import load_workspaces
from backend.mcp import server as srv


@pytest.fixture
def setup(tmp_path):
    base = tmp_path / "GitHub"
    (base / "yeni-repo" / "docs").mkdir(parents=True)
    (base / "yeni-repo" / "README.md").write_text("# Yeni")
    (base / "yeni-repo" / "docs" / "PLAN.md").write_text("# Plan")
    (base / "yeni-repo" / ".venv" / "lib").mkdir(parents=True)
    (base / "yeni-repo" / ".venv" / "lib" / "README.md").write_text("paket")
    (base / "factcheck_probe").mkdir()
    (tmp_path / "gizli").mkdir()
    config = {"factprobe": {"root": str(base / "factcheck_probe")},
              "_workspaces": {"factprobe": {"projects": {"factprobe": "write"}, "project_base": str(base)}}}
    config_file = tmp_path / "knowledge_projects.json"
    config_file.write_text(json.dumps(config))
    scope = load_workspaces(config)["factprobe"].scope
    return {"base": base, "config": config, "config_file": str(config_file), "scope": scope,
            "path": str(tmp_path / "proposals.json"), "tmp": tmp_path}


def test_proposal_is_recorded_but_not_added(setup):
    text = pr.propose(setup["scope"], setup["config"], "yeni-repo", "yeni-repo", "Yeni is", path=setup["path"])
    assert "approve yeni-repo" in text and "2 Markdown belge" in text  # .venv sayilmaz
    saved = pr.load(setup["path"])["yeni-repo"]
    assert saved["root"] == os.path.realpath(setup["base"] / "yeni-repo") and saved["workspace"] == "factprobe"
    assert "yeni-repo" not in json.load(open(setup["config_file"]))


@pytest.mark.parametrize("folder, message", [
    ("../gizli", "gizli klasor"), ("yeni-repo/../../gizli", "gizli klasor"), (".ssh", "gizli klasor"),
    ("/etc", "tam yol"), ("C:\\Users\\x\\.ssh", "tam yol"), ("yok", "bulunamadi"), ("", "bos olamaz"),
    ("factcheck_probe", "zaten bir proje"),
])
def test_folders_outside_base_hidden_missing_or_taken_are_rejected(setup, folder, message):
    with pytest.raises(ValueError, match=message):
        pr.propose(setup["scope"], setup["config"], "x", folder, path=setup["path"])


def test_symlink_out_of_base_is_rejected(setup):
    os.symlink(setup["tmp"] / "gizli", setup["base"] / "kisayol")
    with pytest.raises(ValueError, match="altinda olmali"):
        pr.propose(setup["scope"], setup["config"], "x", "kisayol", path=setup["path"])


def test_name_rules_and_workspace_without_base(setup):
    with pytest.raises(ValueError, match="kucuk harf"):
        pr.propose(setup["scope"], setup["config"], "Yeni Repo", "yeni-repo", path=setup["path"])
    with pytest.raises(ValueError, match="zaten var"):
        pr.propose(setup["scope"], setup["config"], "factprobe", "yeni-repo", path=setup["path"])
    with pytest.raises(ValueError, match="onerilemez"):
        pr.propose(Scope.workspace("w", write=["factprobe"]), setup["config"], "x", "yeni-repo", path=setup["path"])


def test_pending_limit(setup, monkeypatch):
    monkeypatch.setattr(pr, "MAX_PENDING", 1)
    pr.propose(setup["scope"], setup["config"], "yeni-repo", "yeni-repo", path=setup["path"])
    (setup["base"] / "ikinci").mkdir()
    with pytest.raises(ValueError, match="bekleyen onerisi var"):
        pr.propose(setup["scope"], setup["config"], "ikinci", "ikinci", path=setup["path"])


def test_approve_adds_project_to_workspace_and_keeps_backup(setup):
    pr.propose(setup["scope"], setup["config"], "yeni-repo", "yeni-repo", "Yeni is", path=setup["path"])
    pr.approve("yeni-repo", config_file=setup["config_file"], path=setup["path"], home=True)
    config = json.load(open(setup["config_file"]))
    assert config["yeni-repo"]["root"] == os.path.realpath(setup["base"] / "yeni-repo")
    assert ".*/**" in config["yeni-repo"]["exclude"]
    ws = load_workspaces(config)["factprobe"]
    assert ws.scope.can_write("yeni-repo") and ws.scope.default == "yeni-repo"
    assert os.path.exists(setup["config_file"] + ".bak") and pr.load(setup["path"]) == {}
    with pytest.raises(ValueError, match="bekleyen oneri yok"):
        pr.approve("yeni-repo", config_file=setup["config_file"], path=setup["path"])


def test_reject_drops_the_proposal(setup):
    pr.propose(setup["scope"], setup["config"], "yeni-repo", "yeni-repo", path=setup["path"])
    pr.reject("yeni-repo", path=setup["path"])
    assert pr.load(setup["path"]) == {}


def test_tool_is_listed_only_when_workspace_has_a_base(setup):
    class Svc:
        def __init__(self, scope):
            self.scope = scope

    names = lambda scope: {t["name"] for t in srv.MCPServer(Svc(scope)).tools}
    assert "propose_project" in names(setup["scope"])
    assert "propose_project" not in names(Scope.workspace("w", write=["factprobe"]))
    assert "propose_project" not in names(Scope.admin())
