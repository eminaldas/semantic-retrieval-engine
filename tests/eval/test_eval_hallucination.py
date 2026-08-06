from backend.eval import eval_hallucination as harness


class TestLoadGoldenSet:
    def test_reads_jsonl_lines_and_skips_blank_lines(self, tmp_path):
        f = tmp_path / "golden.jsonl"
        f.write_text(
            '{"cause_entity_id": "A::a", "target_entity_id": "B::b", "question": "q1", '
            '"expect_path": true, "expected_chunk_ids": ["ch_0001"]}\n'
            '\n'
            '{"cause_entity_id": "C::c", "target_entity_id": "D::d", "question": "q2", '
            '"expect_path": false, "expected_chunk_ids": ["ch_0002"]}\n',
            encoding="utf-8",
        )
        records = harness.load_golden_set(str(f))
        assert len(records) == 2
        assert records[0]["question"] == "q1"


class TestCrossCheckStructuralVsSemantic:
    def test_both_clean_does_not_diverge(self):
        judgment = harness.HallucinationJudgment(grounded=True, unsupported_claims=[], reasoning="ok")
        result = harness.cross_check_structural_vs_semantic([], judgment)
        assert result == {"structurally_clean": True, "semantically_grounded": True, "diverges": False}

    def test_both_flagged_does_not_diverge(self):
        judgment = harness.HallucinationJudgment(grounded=False, unsupported_claims=["X iddiasi kanitta yok"], reasoning="eksik")
        result = harness.cross_check_structural_vs_semantic(["S3 gecersiz kaynak"], judgment)
        assert result == {"structurally_clean": False, "semantically_grounded": False, "diverges": False}

    def test_structurally_clean_but_semantically_ungrounded_diverges(self):
        # En ilginc durum: atif indeksleri gecerli ama icerik uydurma
        judgment = harness.HallucinationJudgment(grounded=False, unsupported_claims=["uydurma iddia"], reasoning="kanitta yok")
        result = harness.cross_check_structural_vs_semantic([], judgment)
        assert result == {"structurally_clean": True, "semantically_grounded": False, "diverges": True}

    def test_structurally_dirty_but_semantically_grounded_diverges(self):
        judgment = harness.HallucinationJudgment(grounded=True, unsupported_claims=[], reasoning="hepsi destekli")
        result = harness.cross_check_structural_vs_semantic(["S2 used_sources disinda"], judgment)
        assert result == {"structurally_clean": False, "semantically_grounded": True, "diverges": True}


class TestAggregateHallucinationResults:
    def test_all_grounded_gives_rate_one(self):
        results = [{"grounded": True}, {"grounded": True}]
        agg = harness.aggregate_hallucination_results(results)
        assert agg == {"groundedness_rate": 1.0}

    def test_mixed_results_computes_correct_rate(self):
        results = [{"grounded": True}, {"grounded": False}, {"grounded": True}]
        agg = harness.aggregate_hallucination_results(results)
        assert agg == {"groundedness_rate": 2 / 3}

    def test_empty_results_returns_zero_without_error(self):
        agg = harness.aggregate_hallucination_results([])
        assert agg == {"groundedness_rate": 0.0}
