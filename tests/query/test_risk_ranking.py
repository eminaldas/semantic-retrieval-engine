from backend.query import step5c_risk_ranking as risk_ranking


class TestFormatRiskSummary:
    def test_formats_numbered_list_with_scores(self):
        nodes = [
            {"entity_id": "FAILURE_MODE::deadlock", "name": "deadlock", "type": "FAILURE_MODE", "score": 0.42},
            {"entity_id": "DATABASE::postgresql", "name": "postgresql", "type": "DATABASE", "score": 0.31},
        ]
        summary = risk_ranking.format_risk_summary(nodes)
        assert "En yüksek risk skoruna sahip 2 nokta" in summary
        assert "1. FAILURE_MODE::deadlock (skor=0.42)" in summary
        assert "2. DATABASE::postgresql (skor=0.31)" in summary

    def test_empty_list_returns_not_found_message(self):
        summary = risk_ranking.format_risk_summary([])
        assert "bulunamadi" in summary


class TestBuildRiskEvidencePool:
    CHUNKS = {
        "ch_0001": {"chunk_id": "ch_0001", "text": "a"},
        "ch_0002": {"chunk_id": "ch_0002", "text": "b"},
        "ch_0003": {"chunk_id": "ch_0003", "text": "c"},
    }

    def test_merges_node_evidence_and_seed_chunks_without_duplicates(self):
        nodes = [{"entity_id": "X::x", "evidence_chunk_ids": ["ch_0001"]}]
        pool = risk_ranking.build_risk_evidence_pool(nodes, seed_chunk_ids=["ch_0001", "ch_0002"], chunks_by_id=self.CHUNKS)
        assert {c["chunk_id"] for c in pool} == {"ch_0001", "ch_0002"}

    def test_ignores_chunk_ids_missing_from_chunks_by_id(self):
        pool = risk_ranking.build_risk_evidence_pool([], seed_chunk_ids=["ch_9999"], chunks_by_id=self.CHUNKS)
        assert pool == []

    def test_combines_evidence_from_multiple_nodes(self):
        nodes = [
            {"entity_id": "X::x", "evidence_chunk_ids": ["ch_0001"]},
            {"entity_id": "Y::y", "evidence_chunk_ids": ["ch_0003"]},
        ]
        pool = risk_ranking.build_risk_evidence_pool(nodes, seed_chunk_ids=[], chunks_by_id=self.CHUNKS)
        assert {c["chunk_id"] for c in pool} == {"ch_0001", "ch_0003"}
