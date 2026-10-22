from backend.query import query_engine as qe


def rel(type_, confidence, evidence_chunk_ids):
    return {"type": type_, "confidence": confidence, "evidence_chunk_ids": evidence_chunk_ids}


class TestFindCandidatePathsQuery:
    def test_excludes_superseded_relations(self):
        class RecordingSession:
            def __init__(self):
                self.last_query = None

            def run(self, query, **params):
                self.last_query = query
                return []

        session = RecordingSession()
        qe.find_candidate_paths(session, "SERVICE::auth_service", "DATABASE::payment_database", 2)

        assert "SUPERSEDED" in session.last_query


class TestScorePath:
    def test_high_weight_relation_scores_higher_than_low_weight(self):
        high = qe.score_path([rel("CAUSES", 0.9, ["ch_0001"])])
        low = qe.score_path([rel("DEPENDS_ON", 0.9, ["ch_0001"])])
        assert high > low

    def test_shorter_path_scores_higher_than_longer_path_same_relations(self):
        short = qe.score_path([rel("CAUSES", 0.9, ["ch_0001"])])
        long_ = qe.score_path([rel("CAUSES", 0.9, ["ch_0001"]), rel("CAUSES", 0.9, ["ch_0001"])])
        assert short > long_

    def test_higher_confidence_scores_higher(self):
        high_conf = qe.score_path([rel("IMPACTS", 0.95, ["ch_0001"])])
        low_conf = qe.score_path([rel("IMPACTS", 0.6, ["ch_0001"])])
        assert high_conf > low_conf

    def test_more_distinct_evidence_scores_higher(self):
        diverse = qe.score_path([rel("IMPACTS", 0.8, ["ch_0001", "ch_0002"])])
        single = qe.score_path([rel("IMPACTS", 0.8, ["ch_0001"])])
        assert diverse > single


class TestFormatPath:
    def test_formats_nodes_and_relations_in_order(self):
        path = {
            "names": ["Auth Service", "Service Crash", "Gateway"],
            "relationships": [
                {"type": "CAUSES", "confidence": 0.9, "evidence_chunk_ids": ["ch_0004"]},
                {"type": "IMPACTS", "confidence": 0.85, "evidence_chunk_ids": ["ch_0004"]},
            ],
        }
        assert qe.format_path(path) == "[Auth Service] --(CAUSES)--> [Service Crash] --(IMPACTS)--> [Gateway]"


class TestBuildEvidencePool:
    CHUNKS = {
        "ch_0001": {"chunk_id": "ch_0001", "text": "a"},
        "ch_0002": {"chunk_id": "ch_0002", "text": "b"},
        "ch_0003": {"chunk_id": "ch_0003", "text": "c"},
    }

    def test_merges_path_evidence_and_seed_chunks_without_duplicates(self):
        paths = [
            {"relationships": [rel("CAUSES", 0.9, ["ch_0001"])], "score": 5},
        ]
        pool = qe.build_evidence_pool(paths, seed_chunk_ids=["ch_0001", "ch_0002"], chunks_by_id=self.CHUNKS)
        assert {c["chunk_id"] for c in pool} == {"ch_0001", "ch_0002"}

    def test_ignores_chunk_ids_missing_from_chunks_by_id(self):
        pool = qe.build_evidence_pool([], seed_chunk_ids=["ch_9999"], chunks_by_id=self.CHUNKS)
        assert pool == []

    def test_only_considers_top_paths_evidence(self):
        paths = [
            {"relationships": [rel("CAUSES", 0.9, ["ch_0001"])], "score": 5},
            {"relationships": [rel("CAUSES", 0.9, ["ch_0003"])], "score": 1},
        ]
        pool = qe.build_evidence_pool(paths, seed_chunk_ids=[], chunks_by_id=self.CHUNKS, top_paths=1)
        assert {c["chunk_id"] for c in pool} == {"ch_0001"}


class FakeCrossEncoder:
    def __init__(self, score_by_text):
        self.score_by_text = score_by_text

    def predict(self, pairs):
        return [self.score_by_text[text] for _, text in pairs]


class TestRerankEvidence:
    def test_sorts_descending_and_truncates_to_top_n(self):
        pool = [
            {"chunk_id": "ch_0001", "text": "alakasiz"},
            {"chunk_id": "ch_0002", "text": "cok alakali"},
            {"chunk_id": "ch_0003", "text": "orta alakali"},
        ]
        cross_encoder = FakeCrossEncoder({"alakasiz": -5.0, "cok alakali": 9.0, "orta alakali": 2.0})

        ranked = qe.rerank_evidence(cross_encoder, "soru", pool, top_n=2)

        assert [chunk["chunk_id"] for chunk, _ in ranked] == ["ch_0002", "ch_0003"]

    def test_empty_pool_returns_empty_list(self):
        assert qe.rerank_evidence(FakeCrossEncoder({}), "soru", [], top_n=5) == []


class TestBuildContextPack:
    def test_includes_question_path_summary_and_numbered_evidence(self):
        path = {
            "names": ["A", "B"],
            "relationships": [{"type": "CAUSES", "confidence": 0.9, "evidence_chunk_ids": ["ch_0001"]}],
        }
        evidence = [
            ({"chunk_id": "ch_0001", "source_id": "doc.txt", "location": "chunk_index_0", "text": "metin bir"}, 5.0),
            ({"chunk_id": "ch_0002", "source_id": "doc.txt", "location": "chunk_index_1", "text": "metin iki"}, 3.0),
        ]

        pack = qe.build_context_pack("soru?", path, evidence)

        assert pack["question"] == "soru?"
        assert pack["path_summary"] == "[A] --(CAUSES)--> [B]"
        assert "[S1]" in pack["evidence_bundle"] and "metin bir" in pack["evidence_bundle"]
        assert "[S2]" in pack["evidence_bundle"] and "metin iki" in pack["evidence_bundle"]

    def test_handles_missing_path_and_empty_evidence(self):
        pack = qe.build_context_pack("soru?", None, [])
        assert "bulunamadi" in pack["path_summary"]
        assert "bulunamadi" in pack["evidence_bundle"]

    def test_path_summary_override_replaces_best_path_derivation(self):
        pack = qe.build_context_pack("soru?", None, [], path_summary_override="risk siralamasi metni")
        assert pack["path_summary"] == "risk siralamasi metni"


class TestLoadPromptTemplate:
    def test_extracts_fenced_block_and_keeps_placeholders(self, tmp_path):
        md = tmp_path / "prompt.md"
        md.write_text(
            "# Baslik\n\n```\nSystem prompt\nQuestion: {QUESTION}\n```\n",
            encoding="utf-8",
        )
        template = qe.load_prompt_template(str(md))
        assert template == "System prompt\nQuestion: {QUESTION}"


class FakeAnswer:
    def __init__(self, answer_text, used_sources):
        self.answer_text = answer_text
        self.used_sources = used_sources


class TestValidateCitations:
    def test_no_issues_when_citations_and_used_sources_match(self):
        answer = FakeAnswer("Sebep [S1] ve [S2] kanitliyor.", ["S1", "S2"])
        assert qe.validate_citations(answer, available_sids=["S1", "S2", "S3"]) == []

    def test_flags_citation_missing_from_used_sources(self):
        answer = FakeAnswer("Sebep [S1] ve [S2] kanitliyor.", ["S1"])
        issues = qe.validate_citations(answer, available_sids=["S1", "S2"])
        assert len(issues) == 1
        assert "S2" in issues[0]

    def test_flags_used_source_outside_available_range(self):
        answer = FakeAnswer("Sebep [S1] kanitliyor.", ["S1", "S9"])
        issues = qe.validate_citations(answer, available_sids=["S1", "S2"])
        assert len(issues) == 1
        assert "S9" in issues[0]


class TestResolveMode:
    def test_explicit_mode_always_wins(self):
        assert qe.resolve_mode("open", None, None) == "open"
        assert qe.resolve_mode("general", "SERVICE::x", "DATABASE::y") == "general"

    def test_defaults_to_pair_when_cause_or_target_given(self):
        assert qe.resolve_mode(None, "SERVICE::x", None) == "pair"
        assert qe.resolve_mode(None, None, "DATABASE::y") == "pair"
        assert qe.resolve_mode(None, "SERVICE::x", "DATABASE::y") == "pair"

    def test_defaults_to_general_when_nothing_given(self):
        assert qe.resolve_mode(None, None, None) == "general"


class TestDetectModeFromQuestion:
    def test_returns_none_when_no_gemini_client(self):
        assert qe.detect_mode_from_question("soru", None, "model", "template") is None

    def test_returns_detected_mode_on_success(self, monkeypatch):
        monkeypatch.setattr(
            qe.gemini_retry, "call_with_retry", lambda fn, **kwargs: "pair"
        )
        result = qe.detect_mode_from_question("soru", object(), "model", "template")
        assert result == "pair"

    def test_returns_none_on_permanent_failure(self, monkeypatch):
        def raise_runtime_error(fn, **kwargs):
            raise RuntimeError("gemini kalici hata")

        monkeypatch.setattr(qe.gemini_retry, "call_with_retry", raise_runtime_error)
        result = qe.detect_mode_from_question("soru", object(), "model", "template")
        assert result is None


class TestHasSufficientEvidence:
    def test_empty_evidence_is_insufficient(self):
        assert qe.has_sufficient_evidence([], threshold=0.0) is False

    def test_top_score_at_or_above_threshold_is_sufficient(self):
        evidence = [({"chunk_id": "a"}, 5.0), ({"chunk_id": "b"}, 1.0)]
        assert qe.has_sufficient_evidence(evidence, threshold=5.0) is True

    def test_top_score_below_threshold_is_insufficient(self):
        evidence = [({"chunk_id": "a"}, 2.9), ({"chunk_id": "b"}, -3.6)]
        assert qe.has_sufficient_evidence(evidence, threshold=3.0) is False

    def test_only_top_score_matters_not_lower_ranked_scores(self):
        evidence = [({"chunk_id": "a"}, 10.0), ({"chunk_id": "b"}, -100.0)]
        assert qe.has_sufficient_evidence(evidence, threshold=9.0) is True

    def test_default_threshold_is_module_constant(self):
        just_above = [({"chunk_id": "a"}, qe.GENERAL_MODE_RELEVANCE_THRESHOLD + 0.01)]
        just_below = [({"chunk_id": "a"}, qe.GENERAL_MODE_RELEVANCE_THRESHOLD - 0.01)]
        assert qe.has_sufficient_evidence(just_above) is True
        assert qe.has_sufficient_evidence(just_below) is False
