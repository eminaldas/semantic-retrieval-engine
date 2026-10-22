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


def _chunk(i, text, src="p:a.md"):
    return {"chunk_id": f"{src}#{i:04d}", "text": text, "location": f"satır {i * 10}-{i * 10 + 9}"}


class TestWindows:
    def test_neighbor_ids_skip_self_and_below_one(self):
        assert search.neighbor_ids("p:a.md#0002", radius=2) == ["p:a.md#0001", "p:a.md#0003", "p:a.md#0004"]

    def test_window_adds_nearest_neighbors_in_order_within_budget(self):
        chunks = {c["chunk_id"]: c for c in [_chunk(i, f"m{i}" * 10) for i in range(1, 8)]}  # 20'ser karakter
        hit = dict(chunks["p:a.md#0004"], rerank_score=1.0, low_confidence=False)
        [window] = search.build_windows([hit], chunks, window_chars=60)
        assert window["context"] == "\n".join(["m3" * 10, "m4" * 10, "m5" * 10])
        assert window["location"] == "satır 30-59"

    def test_lower_ranked_hit_inside_earlier_window_is_dropped(self):
        chunks = {c["chunk_id"]: c for c in [_chunk(i, "x" * 10) for i in range(1, 5)]}
        best = dict(chunks["p:a.md#0002"], rerank_score=2.0)
        inside = dict(chunks["p:a.md#0003"], rerank_score=1.0)
        other = dict(_chunk(1, "y", src="p:b.md"), rerank_score=0.5)
        windows = search.build_windows([best, inside, other], dict(chunks, **{other["chunk_id"]: other}),
                                       window_chars=100)
        assert [w["path"] if "path" in w else w["chunk_id"] for w in windows] == ["p:a.md#0002", "p:b.md#0001"]

    def test_oversized_hit_goes_alone(self):
        chunks = {c["chunk_id"]: c for c in [_chunk(1, "a" * 50), _chunk(2, "b" * 900)]}
        [window] = search.build_windows([dict(chunks["p:a.md#0002"])], chunks, window_chars=700)
        assert window["context"] == "b" * 900

    def test_unparsed_location_keeps_first(self):
        assert search._merge_location(["bölüm 2", "satır 3-4"]) == "bölüm 2"


class TestFormatResults:
    def _r(self, score, low, text=" metin "):
        return {"path": "docs/a.md", "location": "satır 1-3", "section": "Faz 1", "text": text,
                "rerank_score": score, "low_confidence": low}

    def test_includes_source_location_section(self):
        out = search.format_results([self._r(2.5, False)])
        assert out.startswith("[1] docs/a.md (satır 1-3) · Faz 1 · skor 2.50\nmetin")

    def test_prefers_window_context(self):
        out = search.format_results([dict(self._r(2.5, False), context="once\nmetin\nsonra")])
        assert out.endswith("once\nmetin\nsonra")

    def test_budget_cuts_later_results_but_keeps_first(self):
        out = search.format_results([self._r(2, False, "a" * 300), self._r(1, False, "b" * 300)], max_chars=200)
        assert "a" * 300 in out and "b" * 300 not in out
        assert "1 sonuc daha" in out

    def test_warns_when_all_weak(self):
        assert search.format_results([self._r(-3, True)]).startswith("UYARI")

    def test_no_results(self):
        assert "bulunamadi" in search.format_results([])
