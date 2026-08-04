import scale_test_eval_retrieval as ser


class TestComputeHit:
    def test_hit_when_expected_source_in_retrieved_chunks(self):
        chunks_by_id = {
            "a_ch0001": {"source_id": "a.txt"},
            "b_ch0001": {"source_id": "b.txt"},
        }
        assert ser.compute_hit("a.txt", ["b_ch0001", "a_ch0001"], chunks_by_id) is True

    def test_miss_when_expected_source_not_in_retrieved_chunks(self):
        chunks_by_id = {"b_ch0001": {"source_id": "b.txt"}}
        assert ser.compute_hit("a.txt", ["b_ch0001"], chunks_by_id) is False


class TestAggregateHitRate:
    def test_computes_fraction(self):
        assert ser.aggregate_hit_rate([True, True, False, True]) == 0.75

    def test_empty_list_returns_zero(self):
        assert ser.aggregate_hit_rate([]) == 0.0
