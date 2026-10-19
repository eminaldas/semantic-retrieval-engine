import pytest

from backend.knowledge import projects


class TestIterProjectFiles:
    def test_include_exclude_and_root_level_glob(self, tmp_path):
        (tmp_path / "docs" / "plans").mkdir(parents=True)
        (tmp_path / "docs" / "plans" / "faz1.md").write_text("x", encoding="utf-8")
        (tmp_path / "CLAUDE.md").write_text("x", encoding="utf-8")
        (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
        (tmp_path / "node_modules" / "pkg" / "README.md").write_text("x", encoding="utf-8")
        (tmp_path / "app.tsx").write_text("x", encoding="utf-8")

        found = projects.iter_project_files(str(tmp_path), ["**/*.md"], ["node_modules/**"])

        assert found == ["CLAUDE.md", "docs/plans/faz1.md"]


class TestDiffManifest:
    def test_detects_new_changed_and_removed(self):
        old = {"a.md": "1", "b.md": "2", "c.md": "3"}
        new = {"a.md": "1", "b.md": "X", "d.md": "4"}
        assert projects.diff_manifest(old, new) == (["b.md", "d.md"], ["c.md"])


class TestManifestRoundTrip:
    def test_save_then_load(self, tmp_path):
        projects.save_manifest("nehaber", {"a.md": "h"}, manifest_dir=str(tmp_path))
        assert projects.load_manifest("nehaber", manifest_dir=str(tmp_path)) == {"a.md": "h"}

    def test_missing_manifest_is_empty(self, tmp_path):
        assert projects.load_manifest("yok", manifest_dir=str(tmp_path)) == {}


class TestHeadingPathAt:
    TEXT = "# Forum\n\nGiris\n\n## Faz 1\n\n### Bildirimler\n\nmetin\n\n## Faz 2\n\n```\n# kod yorumu\n```\nson\n"

    def test_nested_headings(self):
        assert projects.heading_path_at(self.TEXT, 9) == ["Forum", "Faz 1", "Bildirimler"]

    def test_sibling_heading_replaces_previous_branch(self):
        assert projects.heading_path_at(self.TEXT, 16) == ["Forum", "Faz 2"]

    def test_before_any_heading(self):
        assert projects.heading_path_at("duz metin\n# Baslik", 1) == []


class TestBuildChunkRecords:
    def test_records_carry_project_section_and_context_prefix(self):
        raw = "# Leaderboard\n\n## Oduller\n\nRozetler haftalik verilir.\n"
        records = projects.build_chunk_records("nehaber", "docs/lb.md", raw,
                                               [("Rozetler haftalik verilir.", "satır 5-5")], mtime=0)
        r = records[0]
        assert r["chunk_id"] == "nehaber:docs/lb.md#0001"
        assert r["source_id"] == "nehaber:docs/lb.md"
        assert r["section"] == "Leaderboard > Oduller"
        assert r["doc_type"] == "doc"
        assert r["embed_text"].startswith("Proje: nehaber | Belge: docs/lb.md | Bölüm: Leaderboard > Oduller\n")
        assert r["text"] == "Rozetler haftalik verilir."

    def test_notes_prefix_marks_doc_type_note(self):
        records = projects.build_chunk_records("nehaber", "notes/x.md", "a", [("a", "satır 1-1")])
        assert records[0]["doc_type"] == "note"


class TestProjectSettings:
    def test_unknown_project_lists_known_ones(self):
        with pytest.raises(KeyError, match="nehaber"):
            projects.project_settings({"nehaber": {"root": "/x"}}, "baska")

    def test_defaults_applied(self):
        s = projects.project_settings({"nehaber": {"root": "/x"}}, "nehaber")
        assert s["include"] == projects.DEFAULT_INCLUDE


class TestNotes:
    def test_note_filename_is_ascii_slug(self):
        from datetime import datetime, timezone

        name = projects.note_filename("Forum Faz 1 kararı: Şikayet akışı", now=datetime(2026, 10, 19, 20, 5, tzinfo=timezone.utc))
        assert name == "2026-10-19-2005-forum-faz-1-karari-sikayet-akisi.md"


class TestSplitOversized:
    def test_small_chunks_untouched(self):
        assert projects.split_oversized([("kisa", "satır 1-1")], max_chars=10) == [("kisa", "satır 1-1")]

    def test_splits_on_paragraphs_then_lines_and_keeps_location(self):
        text = "aaaa\n\nbbbb\n\ncccc\ndddd\neeee"
        parts = projects.split_oversized([(text, "satır 1-9")], max_chars=10)
        assert [p for p, _ in parts] == ["aaaa\n\nbbbb", "cccc\ndddd", "eeee"]
        assert all(loc == "satır 1-9" for _, loc in parts)

    def test_hard_splits_a_single_long_line(self):
        parts = projects.split_oversized([("x" * 25, "satır 1-1")], max_chars=10)
        assert [len(p) for p, _ in parts] == [10, 10, 5]
