from backend.eval import eval_retrieval as harness


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
        assert records[1]["expect_path"] is False


def make_record(expect_path, expected_chunk_ids, question="q"):
    return {
        "cause_entity_id": "A::a", "target_entity_id": "B::b",
        "question": question, "expect_path": expect_path, "expected_chunk_ids": expected_chunk_ids,
    }


def make_evidence(chunk_ids):
    return [({"chunk_id": cid}, 1.0 - i) for i, cid in enumerate(chunk_ids)]


class TestComputeMetrics:
    def test_path_match_true_when_path_found_and_expected(self):
        record = make_record(expect_path=True, expected_chunk_ids=["ch_0001"])
        result = harness.compute_metrics(record, candidate_paths=[{"score": 1.0}], ranked_evidence=make_evidence(["ch_0001"]))
        assert result["path_match"] is True

    def test_path_match_false_when_path_found_but_not_expected(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0001"])
        result = harness.compute_metrics(record, candidate_paths=[{"score": 1.0}], ranked_evidence=make_evidence(["ch_0001"]))
        assert result["path_match"] is False

    def test_path_match_true_when_no_path_and_not_expected(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0001"])
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=make_evidence(["ch_0001"]))
        assert result["path_match"] is True

    def test_top1_hit_true_when_first_evidence_is_expected(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0003"])
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=make_evidence(["ch_0003", "ch_0099"]))
        assert result["top1_hit"] is True

    def test_top1_hit_false_when_first_evidence_is_not_expected(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0003"])
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=make_evidence(["ch_0099", "ch_0003"]))
        assert result["top1_hit"] is False

    def test_top1_hit_false_when_evidence_is_empty(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0003"])
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=[])
        assert result["top1_hit"] is False

    def test_recall_is_one_when_all_expected_found(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0003", "ch_0005"])
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=make_evidence(["ch_0003", "ch_0005", "ch_0099"]))
        assert result["recall"] == 1.0

    def test_recall_is_partial_when_some_expected_found(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0003", "ch_0005"])
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=make_evidence(["ch_0003", "ch_0099"]))
        assert result["recall"] == 0.5

    def test_recall_is_zero_when_none_expected_found(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0003"])
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=make_evidence(["ch_0099"]))
        assert result["recall"] == 0.0

    def test_result_carries_question_and_entity_ids_for_reporting(self):
        record = make_record(expect_path=False, expected_chunk_ids=["ch_0003"], question="Ornek soru?")
        result = harness.compute_metrics(record, candidate_paths=[], ranked_evidence=make_evidence(["ch_0003"]))
        assert result["question"] == "Ornek soru?"
        assert result["cause_entity_id"] == "A::a"
        assert result["target_entity_id"] == "B::b"


def make_result(path_match, top1_hit, recall):
    return {"question": "q", "cause_entity_id": "A::a", "target_entity_id": "B::b",
            "path_match": path_match, "top1_hit": top1_hit, "recall": recall}


class TestAggregateResults:
    def test_all_perfect_gives_rate_one(self):
        results = [make_result(True, True, 1.0), make_result(True, True, 1.0)]
        agg = harness.aggregate_results(results)
        assert agg["path_match_rate"] == 1.0
        assert agg["top1_hit_rate"] == 1.0
        assert agg["mean_recall"] == 1.0

    def test_mixed_results_computes_correct_rates(self):
        results = [
            make_result(True, True, 1.0),
            make_result(False, True, 0.5),
            make_result(True, False, 0.0),
        ]
        agg = harness.aggregate_results(results)
        assert agg["path_match_rate"] == 2 / 3
        assert agg["top1_hit_rate"] == 2 / 3
        assert agg["mean_recall"] == 0.5

    def test_empty_results_returns_zeroes_without_error(self):
        agg = harness.aggregate_results([])
        assert agg == {"path_match_rate": 0.0, "top1_hit_rate": 0.0, "mean_recall": 0.0}
