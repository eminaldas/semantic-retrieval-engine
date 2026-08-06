import numpy as np
import pytest
from backend.pipeline import self_correction as sc


def make_edge(type_, confidence, evidence_chunk_ids, target_id="B", target_name="B", status="ACTIVE"):
    return {
        "type": type_, "confidence": confidence, "evidence_chunk_ids": evidence_chunk_ids,
        "target_id": target_id, "target_name": target_name, "status": status,
    }


def constant_embed_fn(text):
    """Her metne ayni vektoru dondurur - benzerlik filtresini fiilen devre disi
    birakir, testte odak filtre disindaki mantikta olsun diye kullanilir."""
    return np.array([1.0, 0.0])


FAKE_VECTORS = {
    "eski metin benzer": np.array([1.0, 0.0]),
    "yeni metin benzer": np.array([0.9, 0.436]),
    "yeni metin alakasiz": np.array([0.0, 1.0]),
}


def fake_embed_fn(text):
    return FAKE_VECTORS[text]


class TestPairSimilarEdges:
    def test_returns_old_edges_above_threshold(self):
        chunks_by_id = {
            "ch_old": {"chunk_id": "ch_old", "text": "eski metin benzer"},
            "ch_new": {"chunk_id": "ch_new", "text": "yeni metin benzer"},
        }
        old_edges = [make_edge("DEPENDS_ON", 0.9, ["ch_old"], target_id="RECOVERY", target_name="recovery")]
        new_edge = make_edge("CAUSES", 0.8, ["ch_new"], target_id="DBSERVER", target_name="Database Server")

        result = sc.pair_similar_edges(old_edges, new_edge, chunks_by_id, fake_embed_fn, threshold=0.5)
        assert result == old_edges

    def test_excludes_old_edges_below_threshold(self):
        chunks_by_id = {
            "ch_old": {"chunk_id": "ch_old", "text": "eski metin benzer"},
            "ch_new": {"chunk_id": "ch_new", "text": "yeni metin alakasiz"},
        }
        old_edges = [make_edge("USES", 0.9, ["ch_old"])]
        new_edge = make_edge("CAUSES", 0.8, ["ch_new"])

        result = sc.pair_similar_edges(old_edges, new_edge, chunks_by_id, fake_embed_fn, threshold=0.5)
        assert result == []

    def test_missing_chunk_text_treated_as_no_match(self):
        chunks_by_id = {"ch_new": {"chunk_id": "ch_new", "text": "yeni metin benzer"}}
        old_edges = [make_edge("USES", 0.9, ["ch_missing"])]
        new_edge = make_edge("CAUSES", 0.8, ["ch_new"])

        result = sc.pair_similar_edges(old_edges, new_edge, chunks_by_id, fake_embed_fn, threshold=0.5)
        assert result == []

    def test_empty_old_edges_returns_empty(self):
        chunks_by_id = {"ch_new": {"chunk_id": "ch_new", "text": "yeni metin benzer"}}
        new_edge = make_edge("CAUSES", 0.8, ["ch_new"])
        assert sc.pair_similar_edges([], new_edge, chunks_by_id, fake_embed_fn, threshold=0.5) == []


class TestBuildConflictCandidates:
    def test_flags_source_with_old_and_new_evidence_on_different_types_same_target(self):
        rows = [{
            "source_id": "DATABASE::postgresql", "source_name": "PostgreSQL",
            "edges": [
                make_edge("USES", 0.9, ["ch_0001"], target_id="MECHANISM::wal", target_name="WAL"),
                make_edge("PREVENTS", 0.85, ["ch_0009"], target_id="MECHANISM::wal", target_name="WAL"),
            ],
        }]
        chunks_by_id = {
            "ch_0001": {"chunk_id": "ch_0001", "text": "eski"},
            "ch_0009": {"chunk_id": "ch_0009", "text": "yeni"},
        }
        candidates = sc.build_conflict_candidates(rows, {"ch_0009"}, chunks_by_id, constant_embed_fn, similarity_threshold=0.5)
        assert len(candidates) == 1
        assert candidates[0]["source_id"] == "DATABASE::postgresql"
        assert [e["type"] for e in candidates[0]["old_edges"]] == ["USES"]
        assert candidates[0]["new_edge"]["type"] == "PREVENTS"

    def test_flags_source_with_new_edge_pointing_to_a_different_target(self):
        # recovery.conf senaryosu (devam 20): kaynak ayni, hedef VE tip farkli.
        rows = [{
            "source_id": "COMPONENT::recovery_conf", "source_name": "recovery.conf",
            "edges": [
                make_edge("DEPENDS_ON", 0.9, ["ch_old"], target_id="MECHANISM::recovery", target_name="recovery"),
                make_edge("CAUSES", 0.8, ["ch_new"], target_id="SERVICE::database_server", target_name="Database Server"),
            ],
        }]
        chunks_by_id = {
            "ch_old": {"chunk_id": "ch_old", "text": "eski"},
            "ch_new": {"chunk_id": "ch_new", "text": "yeni"},
        }
        candidates = sc.build_conflict_candidates(rows, {"ch_new"}, chunks_by_id, constant_embed_fn, similarity_threshold=0.5)
        assert len(candidates) == 1
        assert candidates[0]["old_edges"][0]["target_name"] == "recovery"
        assert candidates[0]["new_edge"]["target_name"] == "Database Server"

    def test_embedding_filter_excludes_unrelated_coexisting_relation(self):
        rows = [{
            "source_id": "DATABASE::postgresql", "source_name": "PostgreSQL",
            "edges": [
                make_edge("USES", 0.9, ["ch_old"], target_id="MECHANISM::shared_buffer", target_name="shared buffer"),
                make_edge("CAUSES", 0.8, ["ch_new"], target_id="FAILURE_MODE::deadlock", target_name="deadlock"),
            ],
        }]
        chunks_by_id = {
            "ch_old": {"chunk_id": "ch_old", "text": "eski metin benzer"},
            "ch_new": {"chunk_id": "ch_new", "text": "yeni metin alakasiz"},
        }
        candidates = sc.build_conflict_candidates(rows, {"ch_new"}, chunks_by_id, fake_embed_fn, similarity_threshold=0.5)
        assert candidates == []

    def test_splits_a_single_edges_evidence_across_old_and_new(self):
        rows = [{
            "source_id": "A", "source_name": "A",
            "edges": [make_edge("USES", 0.95, ["ch_0003", "ch_0005"], target_id="B", target_name="B")],
        }]
        chunks_by_id = {
            "ch_0003": {"chunk_id": "ch_0003", "text": "eski"},
            "ch_0005": {"chunk_id": "ch_0005", "text": "yeni"},
        }
        candidates = sc.build_conflict_candidates(rows, {"ch_0005"}, chunks_by_id, constant_embed_fn, similarity_threshold=0.5)
        assert len(candidates) == 1
        assert candidates[0]["old_edges"][0]["evidence_chunk_ids"] == ["ch_0003"]
        assert candidates[0]["new_edge"]["evidence_chunk_ids"] == ["ch_0005"]

    def test_still_ignores_source_with_only_old_or_only_new_evidence(self):
        rows = [{
            "source_id": "A", "source_name": "A",
            "edges": [
                make_edge("USES", 0.9, ["ch_0001"], target_id="B", target_name="B"),
                make_edge("USES", 0.8, ["ch_0002"], target_id="B", target_name="B"),
            ],
        }]
        chunks_by_id = {
            "ch_0001": {"chunk_id": "ch_0001", "text": "a"},
            "ch_0002": {"chunk_id": "ch_0002", "text": "b"},
        }
        assert sc.build_conflict_candidates(rows, {"ch_9999"}, chunks_by_id, constant_embed_fn, similarity_threshold=0.5) == []

    def test_ignores_superseded_edges(self):
        rows = [{
            "source_id": "A", "source_name": "A",
            "edges": [
                make_edge("USES", 0.9, ["ch_0001"], target_id="B", target_name="B", status="SUPERSEDED"),
                make_edge("PREVENTS", 0.8, ["ch_0009"], target_id="C", target_name="C"),
            ],
        }]
        chunks_by_id = {
            "ch_0001": {"chunk_id": "ch_0001", "text": "a"},
            "ch_0009": {"chunk_id": "ch_0009", "text": "b"},
        }
        assert sc.build_conflict_candidates(rows, {"ch_0009"}, chunks_by_id, constant_embed_fn, similarity_threshold=0.5) == []


class TestConflictCandidateQuery:
    def test_filters_by_new_chunk_ids_parameter_not_edge_count(self):
        query = sc.build_conflict_candidate_query()
        assert "size(edges) > 1" not in query
        assert "$new_chunk_ids" in query
        assert "ANY(" in query.upper() or "any(" in query

    def test_groups_by_source_only_and_each_edge_carries_its_own_target(self):
        query = sc.build_conflict_candidate_query()
        assert "WITH a, collect(" in query
        assert "target_id: b.entity_id" in query
        assert "target_name: b.name" in query


class TestFormatRelationTypes:
    def test_joins_sorted_distinct_types(self):
        edges = [make_edge("TRIGGERS", 0.9, ["ch_0001"]), make_edge("CAUSES", 0.8, ["ch_0002"])]
        assert sc.format_relation_types(edges) == "CAUSES, TRIGGERS"


class TestFormatEvidenceText:
    def test_includes_relation_type_target_chunk_id_and_text(self):
        edges = [make_edge("USES", 0.9, ["ch_0001"], target_id="MECHANISM::wal", target_name="WAL")]
        chunks_by_id = {"ch_0001": {"chunk_id": "ch_0001", "text": "PostgreSQL WAL kullanir."}}
        result = sc.format_evidence_text(edges, chunks_by_id)
        assert "[USES -> WAL, ch_0001]" in result
        assert "PostgreSQL WAL kullanir." in result

    def test_skips_missing_chunks_silently(self):
        edges = [make_edge("USES", 0.9, ["ch_9999"])]
        assert sc.format_evidence_text(edges, chunks_by_id={}) == ""


class TestBuildPromptVars:
    def test_produces_all_placeholders(self):
        old_edges = [make_edge("USES", 0.9, ["ch_0001"], target_id="MECHANISM::wal", target_name="WAL")]
        new_edge = make_edge("PREVENTS", 0.85, ["ch_0009"], target_id="MECHANISM::wal", target_name="WAL")
        chunks_by_id = {
            "ch_0001": {"chunk_id": "ch_0001", "text": "eski metin"},
            "ch_0009": {"chunk_id": "ch_0009", "text": "yeni metin"},
        }
        variables = sc.build_prompt_vars("PostgreSQL", old_edges, new_edge, chunks_by_id)
        assert variables["{SOURCE_NAME}"] == "PostgreSQL"
        assert variables["{OLD_RELATION_TYPES}"] == "USES"
        assert variables["{NEW_RELATION_TYPE}"] == "PREVENTS"
        assert variables["{NEW_TARGET_NAME}"] == "WAL"
        assert "eski metin" in variables["{OLD_EVIDENCE_TEXT}"]
        assert "yeni metin" in variables["{NEW_EVIDENCE_TEXT}"]


class TestFillPrompt:
    def test_replaces_every_placeholder(self):
        template = "Kaynak: {SOURCE_NAME}, Hedef: {NEW_TARGET_NAME}"
        result = sc.fill_prompt(template, {"{SOURCE_NAME}": "A", "{NEW_TARGET_NAME}": "B"})
        assert result == "Kaynak: A, Hedef: B"


class TestSupersedeOldEdges:
    def test_rejects_relation_type_outside_ontology(self):
        class ExplodingTx:
            def run(self, *args, **kwargs):
                raise AssertionError("Ontoloji disi relation_type Cypher'a hic gonderilmemeli")

        old_edges = [make_edge("DESTROYS", 0.9, ["ch_0001"], target_id="B", target_name="B")]
        with pytest.raises(ValueError, match="Ontoloji disi"):
            sc.supersede_old_edges(ExplodingTx(), "A", old_edges)

    def test_sets_status_superseded_using_each_edges_own_target_id(self):
        calls = []

        class RecordingTx:
            def run(self, query, **params):
                calls.append((query, params))

        old_edges = [
            make_edge("USES", 0.9, ["ch_0001"], target_id="MECHANISM::wal", target_name="WAL"),
            make_edge("TRIGGERS", 0.8, ["ch_0002"], target_id="FAILURE_MODE::deadlock", target_name="deadlock"),
        ]
        sc.supersede_old_edges(RecordingTx(), "DATABASE::postgresql", old_edges)

        assert len(calls) == 2
        for query, params in calls:
            assert "SET r.status = 'SUPERSEDED'" in query
            assert params["source_id"] == "DATABASE::postgresql"
        assert calls[0][1]["target_id"] == "MECHANISM::wal"
        assert calls[1][1]["target_id"] == "FAILURE_MODE::deadlock"
        assert "USES" in calls[0][0]
        assert "TRIGGERS" in calls[1][0]


class TestLoadPromptTemplate:
    def test_extracts_fenced_block(self, tmp_path):
        md = tmp_path / "prompt.md"
        md.write_text("# Baslik\n\n```\nSystem prompt\n\"{SOURCE_NAME}\"\n```\n", encoding="utf-8")
        assert sc.load_prompt_template(str(md)) == 'System prompt\n"{SOURCE_NAME}"'
