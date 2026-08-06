# tests/test_entity_linker.py
import pytest

from backend.query import step5b_entity_linker as linker


class TestLoadPromptTemplate:
    def test_extracts_fenced_block_and_keeps_placeholder(self, tmp_path):
        md = tmp_path / "prompt.md"
        md.write_text(
            "# Baslik\n\naciklama\n\n```\nSystem prompt metni\n\"{QUESTION}\"\n```\n",
            encoding="utf-8",
        )
        template = linker.load_prompt_template(str(md))
        assert template == 'System prompt metni\n"{QUESTION}"'


class FakeQdrantHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class FakeQdrantResponse:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    def __init__(self, hits=None):
        self._hits = hits or []

    def query_points(self, collection_name, query, limit):
        return FakeQdrantResponse(self._hits[:limit])


class FakeEmbeddingModel:
    def encode(self, text):
        return [0.0]  # skor zaten FakeQdrantClient'ta sabit donduruluyor, gercek deger onemsiz


class TestResolvePhraseToEntity:
    def test_accepts_top_hit_above_threshold(self):
        hits = [FakeQdrantHit(0.91, {"entity_id": "SERVICE::auth_servisi", "name": "Auth Servisi", "type": "SERVICE"})]
        qdrant = FakeQdrantClient(hits=hits)
        entity_id, score, candidates = linker.resolve_phrase_to_entity(
            "Auth servisi", qdrant, FakeEmbeddingModel(), threshold=0.85
        )
        assert entity_id == "SERVICE::auth_servisi"
        assert score == 0.91
        assert candidates == [("SERVICE::auth_servisi", "Auth Servisi", "SERVICE", 0.91)]

    def test_rejects_top_hit_below_threshold_but_still_returns_candidates(self):
        hits = [
            FakeQdrantHit(0.62, {"entity_id": "FAILURE_MODE::deadlock", "name": "Deadlock", "type": "FAILURE_MODE"}),
            FakeQdrantHit(0.58, {"entity_id": "FAILURE_MODE::livelock", "name": "Livelock", "type": "FAILURE_MODE"}),
        ]
        qdrant = FakeQdrantClient(hits=hits)
        entity_id, score, candidates = linker.resolve_phrase_to_entity(
            "kilitlenme", qdrant, FakeEmbeddingModel(), threshold=0.85
        )
        assert entity_id is None
        assert score is None
        assert len(candidates) == 2
        assert candidates[0][0] == "FAILURE_MODE::deadlock"

    def test_empty_collection_returns_no_entity_and_no_candidates(self):
        qdrant = FakeQdrantClient(hits=[])
        entity_id, score, candidates = linker.resolve_phrase_to_entity(
            "hicbir sey", qdrant, FakeEmbeddingModel(), threshold=0.85
        )
        assert entity_id is None
        assert score is None
        assert candidates == []

    def test_respects_top_k_limit(self):
        hits = [FakeQdrantHit(0.9 - i * 0.05, {"entity_id": f"SERVICE::e{i}", "name": f"E{i}", "type": "SERVICE"}) for i in range(5)]
        qdrant = FakeQdrantClient(hits=hits)
        _, _, candidates = linker.resolve_phrase_to_entity(
            "x", qdrant, FakeEmbeddingModel(), threshold=0.85, top_k=2
        )
        assert len(candidates) == 2


class FakePhrases:
    def __init__(self, cause_phrase, target_phrase):
        self.cause_phrase = cause_phrase
        self.target_phrase = target_phrase


class TestLinkEntities:
    def test_returns_result_when_both_sides_resolve(self, monkeypatch):
        monkeypatch.setattr(linker, "call_entity_linker", lambda *a, **k: FakePhrases("Auth servisi", "odeme veritabani"))

        def fake_resolve(phrase, *a, **k):
            if phrase == "Auth servisi":
                return "SERVICE::auth_servisi", 0.93, [("SERVICE::auth_servisi", "Auth Servisi", "SERVICE", 0.93)]
            return "DATABASE::odeme_veri_tabani", 0.88, [("DATABASE::odeme_veri_tabani", "Odeme Veri Tabani", "DATABASE", 0.88)]

        monkeypatch.setattr(linker, "resolve_phrase_to_entity", fake_resolve)

        result = linker.link_entities("soru", None, None, None, "model", "template")

        assert result.cause_entity_id == "SERVICE::auth_servisi"
        assert result.cause_phrase == "Auth servisi"
        assert result.cause_score == 0.93
        assert result.target_entity_id == "DATABASE::odeme_veri_tabani"
        assert result.target_score == 0.88

    def test_raises_with_both_failures_reported_when_both_sides_fail(self, monkeypatch):
        monkeypatch.setattr(linker, "call_entity_linker", lambda *a, **k: FakePhrases("belirsiz A", "belirsiz C"))
        monkeypatch.setattr(linker, "resolve_phrase_to_entity", lambda phrase, *a, **k: (None, None, [("SERVICE::x", "X", "SERVICE", 0.4)]))

        with pytest.raises(linker.EntityLinkError) as exc_info:
            linker.link_entities("soru", None, None, None, "model", "template")

        failures = exc_info.value.failures
        assert {f["role"] for f in failures} == {"cause", "target"}
        assert failures[0]["candidates"] == [("SERVICE::x", "X", "SERVICE", 0.4)]

    def test_raises_with_only_failing_side_reported(self, monkeypatch):
        monkeypatch.setattr(linker, "call_entity_linker", lambda *a, **k: FakePhrases("Auth servisi", "belirsiz C"))

        def fake_resolve(phrase, *a, **k):
            if phrase == "Auth servisi":
                return "SERVICE::auth_servisi", 0.93, [("SERVICE::auth_servisi", "Auth Servisi", "SERVICE", 0.93)]
            return None, None, []

        monkeypatch.setattr(linker, "resolve_phrase_to_entity", fake_resolve)

        with pytest.raises(linker.EntityLinkError) as exc_info:
            linker.link_entities("soru", None, None, None, "model", "template")

        failures = exc_info.value.failures
        assert len(failures) == 1
        assert failures[0]["role"] == "target"
        assert failures[0]["phrase"] == "belirsiz C"
