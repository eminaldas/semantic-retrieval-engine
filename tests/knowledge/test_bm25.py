from backend.knowledge import bm25


class TestTokenize:
    def test_lowercases_and_drops_single_chars(self):
        assert bm25.tokenize("Forum Faz 1: a Bildirim") == ["forum", "faz", "bildirim"]

    def test_turkish_capital_i_maps_correctly(self):
        assert bm25.tokenize("IŞIK İzmir") == ["ışık", "izmir"]


class TestDocumentSparseVector:
    def test_indices_sorted_unique_and_aligned_with_values(self):
        indices, values = bm25.document_sparse_vector("leaderboard odul leaderboard kutlama")
        assert indices == sorted(set(indices))
        assert len(indices) == len(values) == 6  # 3 kelime + 3 kok terimi

    def test_repeated_term_weighs_more_but_saturates(self):
        once = dict(zip(*bm25.document_sparse_vector("odul " + "x" * 3)))
        many = dict(zip(*bm25.document_sparse_vector("odul " * 50)))
        tid = bm25.term_id("odul")
        assert many[tid] > once[tid]
        assert many[tid] < bm25.K1 + 1  # BM25 TF doygunluk tavani

    def test_empty_text_gives_empty_vector(self):
        assert bm25.document_sparse_vector("  ") == ([], [])


class TestQuerySparseVector:
    def test_each_unique_term_weight_one_and_stem_half(self):
        indices, values = bm25.query_sparse_vector("odul odul kutlama")
        weights = dict(zip(indices, values))
        assert len(indices) == 4
        assert weights[bm25.term_id("odul")] == 1.0
        assert weights[bm25.term_id(bm25.STEM_PREFIX + bm25.stem("odul"))] == bm25.STEM_QUERY_WEIGHT

    def test_inflected_turkish_words_share_stem_term(self):
        q_idx, _ = bm25.query_sparse_vector("kararları")
        d_idx, _ = bm25.document_sparse_vector("Bu karar emin tarafindan verildi")
        assert bm25.term_id("~karar") in set(q_idx) & set(d_idx)

    def test_query_and_document_share_term_ids(self):
        q_idx, _ = bm25.query_sparse_vector("RewardSheet")
        d_idx, _ = bm25.document_sparse_vector("Kutlama RewardSheet bileseni")
        assert set(q_idx) <= set(d_idx)
