import hashlib
import hmac
import time

from backend.api import dashboard as dash
from backend.query import step5_query_engine as qe


class TestBuildGraphResponse:
    def test_builds_nodes_and_edges_from_neo4j_rows(self):
        node_rows = [
            {"id": "SERVICE::auth_service", "name": "Auth Service", "type": "SERVICE"},
            {"id": "DATABASE::payment_database", "name": "Payment Database", "type": "DATABASE"},
        ]
        edge_rows = [
            {"from_id": "SERVICE::auth_service", "to_id": "DATABASE::payment_database",
             "rel_type": "IMPACTS", "status": "ACTIVE"},
        ]
        result = dash.build_graph_response(node_rows, edge_rows)
        assert result["nodes"] == [
            {"id": "SERVICE::auth_service", "label": "Auth Service", "group": "SERVICE"},
            {"id": "DATABASE::payment_database", "label": "Payment Database", "group": "DATABASE"},
        ]
        assert result["edges"] == [
            {"from": "SERVICE::auth_service", "to": "DATABASE::payment_database",
             "label": "IMPACTS", "status": "ACTIVE"},
        ]

    def test_empty_rows_produce_empty_lists(self):
        assert dash.build_graph_response([], []) == {"nodes": [], "edges": []}


class TestReadSelfCorrectionLog:
    def test_missing_file_returns_empty_list(self, tmp_path):
        missing = tmp_path / "does_not_exist.jsonl"
        assert dash.read_self_correction_log(str(missing)) == []

    def test_parses_jsonl_lines_and_skips_blanks(self, tmp_path):
        log_file = tmp_path / "log.jsonl"
        log_file.write_text(
            '{"source_id": "a", "stage1_classification": "CONTRADICTS"}\n'
            "\n"
            '{"source_id": "b", "stage1_classification": "CONSISTENT"}\n',
            encoding="utf-8",
        )
        entries = dash.read_self_correction_log(str(log_file))
        assert len(entries) == 2
        assert entries[0]["source_id"] == "a"
        assert entries[1]["stage1_classification"] == "CONSISTENT"


class TestBuildPipelineStatus:
    def test_counts_and_distinct_source_files(self):
        chunks = [
            {"chunk_id": "ch_0001", "source_id": "mimari_v1.txt"},
            {"chunk_id": "ch_0002", "source_id": "mimari_v1.txt"},
            {"chunk_id": "ch_0003", "source_id": "other.txt"},
        ]
        result = dash.build_pipeline_status(
            chunks, entity_count=10, relation_count=5,
            qdrant_chunks_count=3, qdrant_entities_count=10,
        )
        assert result == {
            "chunk_count": 3,
            "entity_count": 10,
            "relation_count": 5,
            "qdrant_chunks_points": 3,
            "qdrant_entities_points": 10,
            "source_files": ["mimari_v1.txt", "other.txt"],
        }


class TestBuildPairStructural:
    def test_formats_candidate_paths_and_best_path(self):
        path = {
            "names": ["Auth Service", "Payment Database"],
            "relationships": [{"type": "IMPACTS", "confidence": 0.9, "evidence_chunk_ids": ["ch_0004"]}],
            "score": 5.0,
        }
        structural = dash.build_pair_structural([path], path)
        assert structural == {
            "candidate_paths": [{"score": 5.0, "formatted": "[Auth Service] --(IMPACTS)--> [Payment Database]"}],
            "best_path_formatted": "[Auth Service] --(IMPACTS)--> [Payment Database]",
        }

    def test_no_best_path_when_none_given(self):
        structural = dash.build_pair_structural([], None)
        assert structural == {"candidate_paths": [], "best_path_formatted": None}


class TestBuildOpenStructural:
    def test_formats_ranked_nodes(self):
        ranked_nodes = [
            {"entity_id": "DATABASE::postgresql", "name": "postgresql", "type": "DATABASE", "score": 0.55, "evidence_chunk_ids": ["ch_0001"]},
        ]
        structural = dash.build_open_structural(ranked_nodes)
        assert structural == {"ranked_nodes": [{"entity_id": "DATABASE::postgresql", "score": 0.55}]}

    def test_empty_ranked_nodes(self):
        assert dash.build_open_structural([]) == {"ranked_nodes": []}


class TestBuildQueryResponse:
    def test_general_mode_has_null_structural(self):
        result = dash.build_query_response("general", None, [], None, [], answer_error="GEMINI_API_KEY tanimli degil")
        assert result["mode"] == "general"
        assert result["structural"] is None
        assert result["evidence"] == []
        assert result["answer"] is None
        assert result["answer_error"] == "GEMINI_API_KEY tanimli degil"

    def test_pair_mode_with_structural_and_answer(self):
        path = {
            "names": ["Auth Service", "Payment Database"],
            "relationships": [{"type": "IMPACTS", "confidence": 0.9, "evidence_chunk_ids": ["ch_0004"]}],
            "score": 5.0,
        }
        chunk = {
            "chunk_id": "ch_0004", "source_id": "mimari_v1.txt",
            "location": "p1", "quote": "...", "text": "full text",
        }
        answer = qe.AnswererOutput(
            answer_text="Cevap [S1]", risk_level="HIGH", used_sources=["S1"], assumptions=[]
        )
        structural = dash.build_pair_structural([path], path)
        result = dash.build_query_response("pair", structural, [(chunk, 0.87)], answer, [], answer_error=None)
        assert result["mode"] == "pair"
        assert result["structural"] == {
            "candidate_paths": [{"score": 5.0, "formatted": "[Auth Service] --(IMPACTS)--> [Payment Database]"}],
            "best_path_formatted": "[Auth Service] --(IMPACTS)--> [Payment Database]",
        }
        assert result["evidence"] == [{
            "sid": "S1", "score": 0.87, "chunk_id": "ch_0004", "source_id": "mimari_v1.txt",
            "location": "p1", "quote": "...", "text": "full text",
        }]
        assert result["answer"] == {
            "answer_text": "Cevap [S1]", "risk_level": "HIGH", "used_sources": ["S1"], "assumptions": [],
        }
        assert result["answer_error"] is None

    def test_open_mode_with_structural(self):
        ranked_nodes = [{"entity_id": "DATABASE::postgresql", "name": "postgresql", "type": "DATABASE", "score": 0.55, "evidence_chunk_ids": []}]
        structural = dash.build_open_structural(ranked_nodes)
        result = dash.build_query_response("open", structural, [], None, [], answer_error="Kanit bulunamadi")
        assert result["mode"] == "open"
        assert result["structural"] == {"ranked_nodes": [{"entity_id": "DATABASE::postgresql", "score": 0.55}]}


class TestBuildConversationHydrationResponse:
    def test_gui_turn_uses_stored_response_as_is(self):
        stored_response = {
            "mode": "general", "structural": None, "evidence": [],
            "answer": {"answer_text": "cevap [S1]", "risk_level": "LOW", "used_sources": ["S1"], "assumptions": []},
            "citation_issues": [], "answer_error": None,
        }
        history = [{"turn_id": 1, "mode": "general", "question": "soru", "answer_text": "cevap [S1]", "response": stored_response}]
        result = dash.build_conversation_hydration_response(history)
        assert result == [{"id": 1, "question": "soru", "response": stored_response}]

    def test_cli_turn_synthesizes_minimal_response(self):
        history = [{"turn_id": 2, "mode": "general", "question": "soru2", "answer_text": "cevap2", "response": None}]
        result = dash.build_conversation_hydration_response(history)
        assert result == [{
            "id": 2,
            "question": "soru2",
            "response": {
                "mode": "general", "structural": None, "evidence": [],
                "answer": {"answer_text": "cevap2", "risk_level": None, "used_sources": [], "assumptions": []},
                "citation_issues": [], "answer_error": None,
            },
        }]

    def test_empty_history_returns_empty_list(self):
        assert dash.build_conversation_hydration_response([]) == []


class TestBuildSourcesResponse:
    def test_builds_sorted_list_with_zero_extraction_default(self):
        chunk_groups = {
            "b.txt": {"chunk_count": 1, "ingested_at": "2026-01-02T00:00:00+00:00", "chunk_ids": ["b_ch0001"]},
            "a.txt": {"chunk_count": 2, "ingested_at": "2026-01-01T00:00:00+00:00", "chunk_ids": ["a_ch0001", "a_ch0002"]},
        }
        extraction_groups = {"a.txt": {"entity_count": 3, "relation_count": 1}}
        result = dash.build_sources_response(chunk_groups, extraction_groups)
        assert result == [
            {"source_id": "a.txt", "chunk_count": 2, "entity_count": 3, "relation_count": 1, "ingested_at": "2026-01-01T00:00:00+00:00"},
            {"source_id": "b.txt", "chunk_count": 1, "entity_count": 0, "relation_count": 0, "ingested_at": "2026-01-02T00:00:00+00:00"},
        ]

    def test_empty_groups_produce_empty_list(self):
        assert dash.build_sources_response({}, {}) == []


class TestBuildSourceContentResponse:
    def test_sorts_by_chunk_id_and_selects_fields(self):
        chunks = [
            {"chunk_id": "a_ch0002", "source_id": "a.txt", "location": "chunk_index_1", "text": "ikinci", "quote": "..."},
            {"chunk_id": "a_ch0001", "source_id": "a.txt", "location": "chunk_index_0", "text": "birinci", "quote": "..."},
        ]
        result = dash.build_source_content_response(chunks)
        assert result == [
            {"chunk_id": "a_ch0001", "location": "chunk_index_0", "text": "birinci"},
            {"chunk_id": "a_ch0002", "location": "chunk_index_1", "text": "ikinci"},
        ]

    def test_empty_list(self):
        assert dash.build_source_content_response([]) == []


class TestBuildEntityLinkErrorDetail:
    def test_formats_failures_into_message_and_role_phrase_list(self):
        failures = [
            {"role": "cause", "phrase": "auth servisi", "candidates": [("SERVICE::auth_service", "Auth Service", "SERVICE", 0.4)]},
            {"role": "target", "phrase": "bilinmeyen bir sey", "candidates": []},
        ]
        detail = dash.build_entity_link_error_detail(failures)
        assert detail == {
            "message": "Sorunuzdaki varlıkları tanıyamadım, lütfen daha açık ifade eder misiniz?",
            "failures": [
                {"role": "cause", "phrase": "auth servisi"},
                {"role": "target", "phrase": "bilinmeyen bir sey"},
            ],
        }

    def test_empty_failures_list(self):
        detail = dash.build_entity_link_error_detail([])
        assert detail["failures"] == []


class TestSessionToken:
    def test_valid_token_verifies(self):
        token = dash.create_session_token("gizli-anahtar")
        assert dash.verify_session_token(token, "gizli-anahtar") is True

    def test_wrong_secret_fails(self):
        token = dash.create_session_token("gizli-anahtar")
        assert dash.verify_session_token(token, "baska-anahtar") is False

    def test_malformed_token_fails(self):
        assert dash.verify_session_token("noktasiz-token", "gizli-anahtar") is False

    def test_expired_token_fails(self):
        old_issued_at = str(int(time.time()) - 1000)
        signature = hmac.new(b"gizli-anahtar", old_issued_at.encode(), hashlib.sha256).hexdigest()
        token = f"{old_issued_at}.{signature}"
        assert dash.verify_session_token(token, "gizli-anahtar", max_age_seconds=500) is False

    def test_future_issued_at_fails(self):
        future_issued_at = str(int(time.time()) + 1000)
        signature = hmac.new(b"gizli-anahtar", future_issued_at.encode(), hashlib.sha256).hexdigest()
        token = f"{future_issued_at}.{signature}"
        assert dash.verify_session_token(token, "gizli-anahtar") is False
