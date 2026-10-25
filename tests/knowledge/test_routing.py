from backend.knowledge import routing


def test_hint_when_other_project_clearly_better():
    assert routing.pick_hint({"factprobe": 0.05, "fake-news": 0.9}, "factprobe") == "fake-news"


def test_no_hint_when_margin_too_small():
    assert routing.pick_hint({"factprobe": 0.6, "fake-news": 0.7}, "factprobe", margin=0.2) is None


def test_no_hint_when_other_project_also_weak():
    assert routing.pick_hint({"factprobe": 0.01, "fake-news": 0.2}, "factprobe", margin=0.1, min_score=0.3) is None


def test_no_hint_without_other_projects():
    assert routing.pick_hint({"factprobe": 0.0}, "factprobe") is None


def test_hint_text_names_project_and_scores():
    text = routing.hint_text("fake-news", {"factprobe": 0.05, "fake-news": 0.91}, "factprobe")
    assert "project='fake-news'" in text and "0.91" in text and "0.05" in text


def test_sweep_counts_false_alarms_and_hits():
    rows = [{"project": "a", "scores": {"a": 0.9, "b": 0.1}},
            {"project": "a", "scores": {"a": 0.1, "b": 0.8}},   # yanlis alarm
            {"project": "b", "scores": {"a": 0.1, "b": 0.9}},   # isabet
            {"project": "b", "scores": {"a": 0.7, "b": 0.75}}]  # marj yetmez
    row = routing.sweep(rows, "a", margins=(0.2,), min_score=0.3)[0]
    assert row["false_alarm"] == 0.5 and row["hint_recall"] == 0.5


def test_probe_scores_each_project_with_one_embedding(monkeypatch):
    calls = []

    def fake_hybrid(client, dense, query, flt=None, limit=10):
        calls.append(flt)
        return [{"text": "x", "project": "?"}] if len(calls) == 1 else []

    class Embedder:
        n = 0

        def encode(self, q):
            Embedder.n += 1
            return [0.0]

    class CrossEncoder:
        def predict(self, pairs):
            return [0.7 for _ in pairs]

    monkeypatch.setattr(routing.store, "hybrid_search", fake_hybrid)
    scores = routing.probe(None, Embedder(), CrossEncoder(), "soru", ["a", "b"])
    assert scores == {"a": 0.7, "b": 0.0} and Embedder.n == 1 and len(calls) == 2


def test_project_hint_survives_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("qdrant yok")

    monkeypatch.setattr(routing, "probe", boom)
    assert routing.project_hint(None, None, None, "soru", "a", ["a", "b"]) is None


def test_golden_files_skip_backups_and_candidates(tmp_path):
    import os

    for name in ("p.jsonl", "p_hard.jsonl", "p_candidates.jsonl", "p_v1_2026-10-25.jsonl", "q.jsonl"):
        (tmp_path / name).write_text("")
    assert [os.path.basename(p) for p in routing.golden_files("p", str(tmp_path))] == ["p.jsonl", "p_hard.jsonl"]


def test_project_hint_reuses_search_score_and_probes_only_others(monkeypatch):
    seen = {}

    def fake_probe(client, embedder, cross_encoder, query, projects):
        seen["projects"] = projects
        return {"b": 0.9}

    monkeypatch.setattr(routing, "probe", fake_probe)
    text = routing.project_hint(None, None, None, "soru", "a", ["a", "b"], current_score=0.1)
    assert seen["projects"] == ["b"] and "project='b'" in text and "0.10" in text


def test_project_hint_skips_probe_when_search_is_confident(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("yoklama yapilmamali")

    monkeypatch.setattr(routing, "probe", fail)
    assert routing.project_hint(None, None, None, "soru", "a", ["a", "b"], current_score=0.85) is None
