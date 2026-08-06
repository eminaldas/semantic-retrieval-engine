from backend.management import manage_sources


class TestGroupChunksBySource:
    def test_groups_and_counts_by_source(self):
        chunks = [
            {"chunk_id": "a_ch0001", "source_id": "a.txt", "ingested_at": "2026-01-01T00:00:00+00:00"},
            {"chunk_id": "a_ch0002", "source_id": "a.txt", "ingested_at": "2026-01-01T00:00:00+00:00"},
            {"chunk_id": "b_ch0001", "source_id": "b.txt", "ingested_at": "2026-01-02T00:00:00+00:00"},
        ]
        groups = manage_sources.group_chunks_by_source(chunks)
        assert groups["a.txt"]["chunk_count"] == 2
        assert groups["a.txt"]["chunk_ids"] == ["a_ch0001", "a_ch0002"]
        assert groups["b.txt"]["chunk_count"] == 1

    def test_missing_ingested_at_becomes_none(self):
        chunks = [{"chunk_id": "a_ch0001", "source_id": "a.txt"}]
        groups = manage_sources.group_chunks_by_source(chunks)
        assert groups["a.txt"]["ingested_at"] is None


class TestGroupExtractionBySource:
    def test_sums_entities_and_relations_per_source(self):
        chunks = [
            {"chunk_id": "a_ch0001", "source_id": "a.txt"},
            {"chunk_id": "b_ch0001", "source_id": "b.txt"},
        ]
        extraction_records = [
            {"chunk_id": "a_ch0001", "entities": [1, 2], "relations": [1]},
            {"chunk_id": "b_ch0001", "entities": [1], "relations": []},
        ]
        groups = manage_sources.group_extraction_by_source(chunks, extraction_records)
        assert groups["a.txt"] == {"entity_count": 2, "relation_count": 1}
        assert groups["b.txt"] == {"entity_count": 1, "relation_count": 0}

    def test_ignores_extraction_records_with_unknown_chunk_id(self):
        chunks = [{"chunk_id": "a_ch0001", "source_id": "a.txt"}]
        extraction_records = [{"chunk_id": "orphan_ch9999", "entities": [1], "relations": []}]
        groups = manage_sources.group_extraction_by_source(chunks, extraction_records)
        assert groups == {}


class TestChunkIdsForSource:
    def test_returns_only_matching_source(self):
        chunks = [
            {"chunk_id": "a_ch0001", "source_id": "a.txt"},
            {"chunk_id": "b_ch0001", "source_id": "b.txt"},
            {"chunk_id": "a_ch0002", "source_id": "a.txt"},
        ]
        assert manage_sources.chunk_ids_for_source(chunks, "a.txt") == ["a_ch0001", "a_ch0002"]

    def test_returns_empty_list_for_unknown_source(self):
        chunks = [{"chunk_id": "a_ch0001", "source_id": "a.txt"}]
        assert manage_sources.chunk_ids_for_source(chunks, "nope.txt") == []


class TestPruneEvidenceChunkIds:
    def test_removes_only_the_specified_ids(self):
        result = manage_sources.prune_evidence_chunk_ids(
            ["a_ch0001", "b_ch0001", "a_ch0002"], ["a_ch0001", "a_ch0002"]
        )
        assert result == ["b_ch0001"]

    def test_keeps_all_when_nothing_matches(self):
        result = manage_sources.prune_evidence_chunk_ids(["b_ch0001"], ["a_ch0001"])
        assert result == ["b_ch0001"]

    def test_empty_result_when_everything_removed(self):
        result = manage_sources.prune_evidence_chunk_ids(["a_ch0001"], ["a_ch0001"])
        assert result == []
