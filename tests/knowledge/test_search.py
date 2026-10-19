from backend.knowledge import search


class FakeCrossEncoder:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]  # uzun metin = yuksek skor


class TestRerank:
    def test_orders_by_score_and_truncates(self):
        pool = [{"text": "a"}, {"text": "ccc"}, {"text": "bb"}]
        ranked = search.rerank(FakeCrossEncoder(), "q", pool, top_k=2)
        assert [r["text"] for r in ranked] == ["ccc", "bb"]
        assert ranked[0]["rerank_score"] == 3.0

    def test_empty_pool(self):
        assert search.rerank(FakeCrossEncoder(), "q", [], 5) == []


class TestFormatResults:
    def _r(self, score, low):
        return {"path": "docs/a.md", "location": "satır 1-3", "section": "Faz 1", "text": " metin ",
                "rerank_score": score, "low_confidence": low}

    def test_includes_source_location_section(self):
        out = search.format_results([self._r(2.5, False)])
        assert out.startswith("[1] docs/a.md (satır 1-3) · Faz 1 · skor 2.50\nmetin")

    def test_warns_when_all_weak(self):
        assert search.format_results([self._r(-3, True)]).startswith("UYARI")

    def test_no_results(self):
        assert "bulunamadi" in search.format_results([])
